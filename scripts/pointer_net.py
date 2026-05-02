"""
Pointer Network architectures for irregular 2D bin-packing via RL.

Two variants are provided:

SplitPointerNet (default)
    Separate board CNN and 3-channel part CNN. Board encoded once per step.
    Each candidate part × rotation is encoded from (delta, result, isolated_part_sdf)
    triple images. All N×R (part, rotation) pairs are scored jointly by the same MLP,
    returning a flat (N, R) logit tensor. Sample from Categorical(logits=joint.reshape(N*R)).
    Trains end-to-end from random init — no pretrained backbone required.

SpatialPointerNet
    Replaces the compressed 128-d per-part embedding with 64 spatial tokens
    (8×8 feature map). All N×64 tokens pass through a shared Transformer that
    attends across both spatial positions and candidates simultaneously.

Episode loop (SplitPointerNet with rotation)
---------------
    env.reset(lib_ids)
    rotations_rad = env.get_rotation_angles(n_rotations=8)
    for step in range(n_parts):
        obs    = torch.from_numpy(env.preview_images_per_rotation(rotations_rad))  # (N, 2R+1, H, W)
        mask   = build_remaining_mask(env.remaining_item_ids(), N)
        joint_logits = model(obs, mask)                           # (N, R)
        action_id = Categorical(logits=joint_logits.reshape(N*R)).sample()
        part_id, rot_id = action_id // R, action_id % R
        env.place_with_rotation(part_id.item(), rotations_rad[rot_id.item()])
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cross-part context Transformer
# ---------------------------------------------------------------------------

class PartTransformer(nn.Module):
    """
    Self-attention over the set of remaining parts.

    Enriches each part's embedding with context from the others still waiting
    to be placed. Placed parts are masked out (src_key_padding_mask).
    Pre-norm, 2-layer, no positional encoding (parts form a set, not a sequence).
    """

    def __init__(self, dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(self, P: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        P    : (N, 128)  part embeddings (all parts, including placed)
        mask : (N,)      1.0 = available, 0.0 = already placed
        Returns (N, 128) context-enriched embeddings.
        """
        padding_mask = (mask == 0).unsqueeze(0)          # (1, N)
        out = self.encoder(P.unsqueeze(0), src_key_padding_mask=padding_mask)
        return out.squeeze(0)                             # (N, 128)


# ---------------------------------------------------------------------------
# RotationHead
# ---------------------------------------------------------------------------

class RotationHead(nn.Module):
    """
    Scores R rotation candidates for a selected part.

    Input:  ctx       (dim,)    context vector for the selected part
            rot_feats (R, dim)  per-rotation CNN embeddings
    Output: rot_logits (R,)
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(True),
            nn.Linear(dim, 1),
        )

    def forward(self, ctx: torch.Tensor, rot_feats: torch.Tensor) -> torch.Tensor:
        """
        ctx      : (dim,)
        rot_feats: (R, dim)
        Returns  : (R,) logits
        """
        R       = rot_feats.shape[0]
        ctx_exp = ctx.unsqueeze(0).expand(R, -1)                      # (R, dim)
        return self.mlp(torch.cat([ctx_exp, rot_feats], dim=-1)).squeeze(-1)  # (R,)


# ---------------------------------------------------------------------------
# SplitPointerNet
# ---------------------------------------------------------------------------

class _SpatialCNN(nn.Module):
    """
    CNN with 2×2 adaptive avg-pool instead of global average pooling.

    Preserves coarse positional structure (four quadrants) that GAP discards.

    Input:  (B, in_channels, 128, 128)
    Output: (B, 128)
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,          64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,          128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,         128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(2),  # (B, 128, 2, 2)
        )
        self.proj = nn.Sequential(
            nn.Flatten(),             # (B, 512)
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class SplitPointerNet(nn.Module):
    """
    Pointer Network with separate board and part encoders. Trains from scratch.

    Scores all N×R (part, rotation) pairs jointly via a Transformer that attends
    over all N×R tokens — enabling the model to reason about interactions between
    specific (part_i, rot_r) and (part_j, rot_s) pairs.

    Per-step forward (R rotations, N parts):
        obs = env.preview_images_per_rotation(rotations_rad)  # (N, 2R+1, 128, 128)
        For each (part i, rotation r):
          delta[i,r]     = obs[i, r+1] − obs[i, 0]
          result[i,r]    = obs[i, r+1]
          isolated[i,r]  = obs[i, R+1+r]
        part_in    = stack triples                             →  (N*R, 3, 128, 128)
        board_feat = board_cnn(board)                          →  (1, 128)
        rot_feats  = part_cnn(part_in).reshape(N, R, 128)     →  (N, R, 128)
        fused      = fusion(cat[board_feat_exp, rot_feats])    →  (N*R, 128)
        ctx        = PartTransformer(fused, mask_NR)           →  (N*R, 128)  ← attends over N×R
        joint_logits = joint_head(ctx).reshape(N, R)           →  (N, R)
    """

    def __init__(self):
        super().__init__()
        self.board_cnn  = _SpatialCNN(in_channels=1)
        self.part_cnn   = _SpatialCNN(in_channels=3)
        self.fusion     = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True))
        self.part_ctx   = PartTransformer()
        self.joint_head = nn.Linear(128, 1)

    def forward(
        self,
        obs:  torch.Tensor,  # (N, 2R+1, 128, 128)
        mask: torch.Tensor,  # (N,) 1=available, 0=placed
    ) -> torch.Tensor:
        """
        Score all N×R (part, rotation) pairs jointly.

        Returns:
            joint_logits : (N, R)   placed parts have all R logits set to −1e9
        """
        N, two_Rp1, H, W = obs.shape
        R = (two_Rp1 - 1) // 2

        board         = obs[0, 0].unsqueeze(0).unsqueeze(0)            # (1, 1, H, W)
        current_flat  = obs[:, 0].unsqueeze(1).expand(N, R, H, W)\
                            .reshape(N * R, 1, H, W)                   # (N*R, 1, H, W)
        after_flat    = obs[:, 1:R + 1].reshape(N * R, 1, H, W)       # (N*R, 1, H, W)
        isolated_flat = obs[:, R + 1:2 * R + 1].reshape(N * R, 1, H, W)  # (N*R, 1, H, W)
        part_in       = torch.cat([after_flat - current_flat,
                                   after_flat,
                                   isolated_flat], dim=1)              # (N*R, 3, H, W)

        board_feat = self.board_cnn(board)                             # (1, 128)
        rot_feats  = self.part_cnn(part_in).reshape(N, R, 128)        # (N, R, 128)

        # Fuse board context into every (part, rotation) token
        board_exp = board_feat.expand(N * R, -1)                       # (N*R, 128)
        fused     = self.fusion(torch.cat([
            board_exp, rot_feats.reshape(N * R, 128),
        ], dim=-1))                                                     # (N*R, 128)

        # Transformer attends over all N×R tokens jointly
        mask_flat    = mask.unsqueeze(1).expand(N, R).reshape(N * R)  # (N*R,)
        ctx          = self.part_ctx(fused, mask_flat)                 # (N*R, 128)
        joint_logits = self.joint_head(ctx).reshape(N, R)             # (N, R)

        # Mask placed parts: all R logits → -1e9
        mask_exp = mask.unsqueeze(1).expand(N, R)                      # (N, R)
        return joint_logits.masked_fill(mask_exp == 0, -1e9)           # (N, R)


# ---------------------------------------------------------------------------
# SpatialPointerNet
# ---------------------------------------------------------------------------

class _SpatialCNNNoPool(nn.Module):
    """
    CNN backbone that preserves the full 8×8 spatial feature map.

    Input:  (B, in_channels, 128, 128)
    Output: (B, 128, 8, 8)
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,          64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,          128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,         128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 128, 8, 8)


class SpatialPointerNet(nn.Module):
    """
    Pointer Network with full spatial reasoning via cross-candidate attention.

    Replaces the compressed 128-d per-part embedding with 64 spatial tokens
    (8×8 feature map) per candidate. All N×64 tokens pass through a shared
    Transformer that attends across both spatial positions and candidates.

    Per-step forward:
        env.preview_pair_images()  →  pair_images  (N, 2, 128, 128)
        board  = pair_images[0, 0]                          →  (1, 1, 128, 128)
        deltas = pair_images[:, 1] − pair_images[:, 0]     →  (N, 1, 128, 128)
        result = pair_images[:, 1]                         →  (N, 1, 128, 128)
        part_in = cat([deltas, result], dim=1)             →  (N, 2, 128, 128)
        board_map  = board_cnn(board)   → (1, 128, 8, 8)  → (1, 64, 128) board tokens
        part_maps  = part_cnn(part_in)  → (N, 128, 8, 8)  → (N, 64, 128) part tokens
        tokens = part_tokens + board_tokens + pos_embed    →  (N, 64, 128)
        tokens → reshape (1, N×64, 128) → SpatialTransformer → reshape (N, 64, 128)
        mean pool over 64 spatial tokens                   →  (N, 128)
        part_head                                          →  (N,)
    """

    def __init__(self):
        super().__init__()
        self.board_cnn   = _SpatialCNNNoPool(in_channels=1)
        self.part_cnn    = _SpatialCNNNoPool(in_channels=2)
        self.pos_embed   = nn.Parameter(torch.randn(64, 128) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False,
        )
        self.part_head = nn.Linear(128, 1)

    def forward(
        self,
        pair_images: torch.Tensor,  # (N, 2, 128, 128)
        mask:        torch.Tensor,  # (N,) 1=available, 0=placed
    ) -> torch.Tensor:
        N = pair_images.shape[0]

        board   = pair_images[0, 0].unsqueeze(0).unsqueeze(0)           # (1, 1, 128, 128)
        deltas  = (pair_images[:, 1] - pair_images[:, 0]).unsqueeze(1)  # (N, 1, 128, 128)
        result  = pair_images[:, 1].unsqueeze(1)                        # (N, 1, 128, 128)
        part_in = torch.cat([deltas, result], dim=1)                    # (N, 2, 128, 128)

        board_map = self.board_cnn(board)    # (1, 128, 8, 8)
        part_maps = self.part_cnn(part_in)   # (N, 128, 8, 8)

        board_tokens = board_map.permute(0, 2, 3, 1).reshape(1, 64, 128)  # (1, 64, 128)
        part_tokens  = part_maps.permute(0, 2, 3, 1).reshape(N, 64, 128)  # (N, 64, 128)

        tokens = part_tokens + board_tokens + self.pos_embed             # (N, 64, 128)
        tokens = tokens.reshape(1, N * 64, 128)                          # (1, N*64, 128)

        token_mask = (mask == 0).unsqueeze(1).expand(N, 64).reshape(1, N * 64)
        tokens = self.spatial_transformer(tokens, src_key_padding_mask=token_mask)

        part_embs = tokens.reshape(N, 64, 128).mean(dim=1)              # (N, 128)
        logits    = self.part_head(part_embs).squeeze(-1)                # (N,)
        return logits.masked_fill(mask == 0, -1e9)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_PARTS     = 6
    N_ROTATIONS = 8
    device      = torch.device("cpu")

    # SplitPointerNet — flat joint (N, R) interface
    model = SplitPointerNet().eval()
    obs   = torch.rand(N_PARTS, 2 * N_ROTATIONS + 1, 128, 128)
    mask  = torch.ones(N_PARTS)
    mask[2] = 0.0

    joint_logits = model(obs, mask)
    assert joint_logits.shape == (N_PARTS, N_ROTATIONS), f"joint_logits: {tuple(joint_logits.shape)}"
    assert (joint_logits[2] == -1e9).all(), "masked part should have all -1e9 logits"
    # Sample flat action
    from torch.distributions import Categorical
    flat = joint_logits.reshape(N_PARTS * N_ROTATIONS)
    action_id = Categorical(logits=flat).sample()
    part_id, rot_id = action_id // N_ROTATIONS, action_id % N_ROTATIONS
    print(f"SplitPointerNet shape check OK — joint_logits: {tuple(joint_logits.shape)}, "
          f"sampled part={part_id.item()} rot={rot_id.item()}")

    # SpatialPointerNet — unchanged interface (returns (N,) logits directly)
    spatial = SpatialPointerNet().eval()
    pair_images = torch.rand(N_PARTS, 2, 128, 128)
    logits = spatial(pair_images, mask)
    assert logits.shape == (N_PARTS,), f"expected ({N_PARTS},), got {tuple(logits.shape)}"
    assert logits[2].item() == -1e9, "masked part should be -1e9"
    print(f"SpatialPointerNet shape check OK — logits: {tuple(logits.shape)}")

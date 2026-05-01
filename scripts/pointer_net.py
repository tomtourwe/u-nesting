"""
Pointer Network architectures for irregular 2D bin-packing via RL.

Two variants are provided:

SplitPointerNet (default)
    Separate board CNN and 2-channel part CNN. Board encoded once per step.
    Each candidate part × rotation is encoded from (delta, result) pair images.
    Returns (part_logits, rot_feats, ctx) for hierarchical (part, rotation) sampling.
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
        obs    = torch.from_numpy(env.preview_images_per_rotation(rotations_rad))
        mask   = build_remaining_mask(env.remaining_item_ids(), N)
        part_logits, rot_feats, ctx, value = model(obs, mask)  # (N,), (N,R,128), (N,128), ()
        part_id  = Categorical(logits=part_logits).sample()
        rot_logits = model.rotation_head(ctx[part_id], rot_feats[part_id])  # (R,)
        rot_id   = Categorical(logits=rot_logits).sample()
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

    Per-step forward (R rotations, N parts):
        obs = env.preview_images_per_rotation(rotations_rad)  # (N, R+1, 128, 128)
        board      = obs[0, 0]                                 →  (1, 1, 128, 128)
        For each (part i, rotation r):
          delta[i,r]  = obs[i, r+1] − obs[i, 0]              →  (1, 128, 128)
          result[i,r] = obs[i, r+1]                           →  (1, 128, 128)
        part_in    = stack all (delta, result) pairs           →  (N*R, 2, 128, 128)
        board_feat = board_cnn(board)                          →  (1, 128)
        rot_feats  = part_cnn(part_in).reshape(N, R, 128)     →  (N, R, 128)
        part_feats = rot_feats.mean(dim=1)                     →  (N, 128)
        fused      = fusion(cat[board_feat, part_feats])       →  (N, 128)
        ctx        = PartTransformer(fused, mask)              →  (N, 128)
        part_logits = part_head(ctx)                           →  (N,)
        # After sampling part_id:
        rot_logits = rotation_head(ctx[part_id], rot_feats[part_id])  →  (R,)
    """

    def __init__(self):
        super().__init__()
        self.board_cnn     = _SpatialCNN(in_channels=1)
        self.part_cnn      = _SpatialCNN(in_channels=2)
        self.fusion        = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True))
        self.part_ctx      = PartTransformer()
        self.part_head     = nn.Linear(128, 1)
        self.rotation_head = RotationHead()
        self.value_head    = nn.Linear(128, 1)

    def rotation_feats_for_part(self, obs_part: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation features for a single part without a full forward pass.

        Used in two-phase eval: after the part head selects a part using cheap
        pair-image observations, this encodes the R rotation candidates for the
        chosen part only.

        obs_part : (R+1, H, W) — channel 0 = board SDF, channels 1..R = after-SDFs
        Returns  : (R, 128)
        """
        R, H, W  = obs_part.shape[0] - 1, obs_part.shape[1], obs_part.shape[2]
        current  = obs_part[0].unsqueeze(0).expand(R, H, W).unsqueeze(1)   # (R, 1, H, W)
        after    = obs_part[1:].unsqueeze(1)                                 # (R, 1, H, W)
        part_in  = torch.cat([after - current, after], dim=1)               # (R, 2, H, W)
        return self.part_cnn(part_in)                                        # (R, 128)

    def forward(
        self,
        obs:  torch.Tensor,  # (N, R+1, 128, 128)
        mask: torch.Tensor,  # (N,) 1=available, 0=placed
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Score all candidate parts jointly and return per-rotation embeddings.

        Returns:
            part_logits : (N,)        placed parts set to −1e9
            rot_feats   : (N, R, 128) per-(part, rotation) CNN embeddings
            ctx         : (N, 128)    context-enriched part embeddings
            value       : ()          scalar state value estimate V(s)
        """
        N, Rp1, H, W = obs.shape
        R = Rp1 - 1

        board        = obs[0, 0].unsqueeze(0).unsqueeze(0)             # (1, 1, H, W)
        current_flat = obs[:, 0].unsqueeze(1).expand(N, R, H, W)\
                           .reshape(N * R, 1, H, W)                    # (N*R, 1, H, W)
        after_flat   = obs[:, 1:].reshape(N * R, 1, H, W)             # (N*R, 1, H, W)
        part_in      = torch.cat([after_flat - current_flat,
                                  after_flat], dim=1)                  # (N*R, 2, H, W)

        board_feat = self.board_cnn(board)                             # (1, 128)
        rot_feats  = self.part_cnn(part_in).reshape(N, R, 128)        # (N, R, 128)
        part_feats = rot_feats.mean(dim=1)                             # (N, 128)

        fused = self.fusion(torch.cat([
            board_feat.expand(N, -1),
            part_feats,
        ], dim=-1))                                                     # (N, 128)

        ctx    = self.part_ctx(fused, mask)                            # (N, 128)
        logits = self.part_head(ctx).squeeze(-1)                       # (N,)

        remaining_ctx = ctx[mask > 0.5]
        if remaining_ctx.shape[0] == 0:
            remaining_ctx = ctx
        value = self.value_head(remaining_ctx.mean(0)).squeeze()       # scalar

        return logits.masked_fill(mask == 0, -1e9), rot_feats, ctx, value


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

    # SplitPointerNet — hierarchical (part, rotation) interface
    model = SplitPointerNet().eval()
    obs   = torch.rand(N_PARTS, N_ROTATIONS + 1, 128, 128)
    mask  = torch.ones(N_PARTS)
    mask[2] = 0.0

    part_logits, rot_feats, ctx, value = model(obs, mask)
    assert part_logits.shape == (N_PARTS,),              f"part_logits: {tuple(part_logits.shape)}"
    assert rot_feats.shape   == (N_PARTS, N_ROTATIONS, 128), f"rot_feats: {tuple(rot_feats.shape)}"
    assert ctx.shape         == (N_PARTS, 128),          f"ctx: {tuple(ctx.shape)}"
    assert part_logits[2].item() == -1e9,                "masked part should be -1e9"
    assert value.shape       == (),                      f"value: {tuple(value.shape)}"

    rot_logits = model.rotation_head(ctx[0], rot_feats[0])
    assert rot_logits.shape == (N_ROTATIONS,),           f"rot_logits: {tuple(rot_logits.shape)}"
    print(f"SplitPointerNet shape check OK — part_logits: {tuple(part_logits.shape)}, "
          f"rot_feats: {tuple(rot_feats.shape)}, rot_logits: {tuple(rot_logits.shape)}, "
          f"value: scalar")

    # SpatialPointerNet — unchanged interface (returns (N,) logits directly)
    spatial = SpatialPointerNet().eval()
    pair_images = torch.rand(N_PARTS, 2, 128, 128)
    logits = spatial(pair_images, mask)
    assert logits.shape == (N_PARTS,), f"expected ({N_PARTS},), got {tuple(logits.shape)}"
    assert logits[2].item() == -1e9, "masked part should be -1e9"
    print(f"SpatialPointerNet shape check OK — logits: {tuple(logits.shape)}")

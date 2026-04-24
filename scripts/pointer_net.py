"""
Pointer Network architectures for irregular 2D bin-packing via RL.

Two variants are provided:

SplitPointerNet (default)
    Separate board CNN and 2-channel part CNN. Board encoded once per step.
    Each candidate part is encoded from (delta, resulting-board) pair images.
    Trains end-to-end from random init — no pretrained backbone required.

SpatialPointerNet
    Replaces the compressed 128-d per-part embedding with 64 spatial tokens
    (8×8 feature map). All N×64 tokens pass through a shared Transformer that
    attends across both spatial positions and candidates simultaneously.

Episode loop (both models)
---------------
    env.reset(lib_ids)
    for step in range(n_parts):
        pair_images_t = torch.from_numpy(env.preview_pair_images())  # (N, 2, 128, 128)
        mask          = build_remaining_mask(env.remaining_item_ids(), N)
        logits        = model(pair_images_t, mask)                   # (N,)
        part_id       = Categorical(logits=logits).sample()
        env.place_anywhere(part_id.item())
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

    Per-step forward:
        env.preview_pair_images()  →  pair_images  (N, 2, 128, 128)
        board    = pair_images[0, 0]                          →  (1, 1, 128, 128)
        deltas   = pair_images[:, 1] − pair_images[:, 0]     →  (N, 1, 128, 128)
        result   = pair_images[:, 1]                         →  (N, 1, 128, 128)
        part_in  = cat([deltas, result], dim=1)              →  (N, 2, 128, 128)
        board_feat = board_cnn(board)                        →  (1, 128)   once per step
        part_feats = part_cnn(part_in)                       →  (N, 128)   one per candidate
        fused      = fusion(cat[board_feat, part_feats])     →  (N, 128)
        ctx        = PartTransformer(fused, mask)            →  (N, 128)
        logits     = part_head(ctx)                          →  (N,)
    """

    def __init__(self):
        super().__init__()
        self.board_cnn = _SpatialCNN(in_channels=1)
        self.part_cnn  = _SpatialCNN(in_channels=2)
        self.fusion    = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
        )
        self.part_ctx  = PartTransformer()
        self.part_head = nn.Linear(128, 1)

    def forward(
        self,
        pair_images: torch.Tensor,  # (N, 2, 128, 128)
        mask:        torch.Tensor,  # (N,) 1=available, 0=placed
    ) -> torch.Tensor:
        """
        Score all candidate parts jointly.

        Returns part_logits (N,) — placed parts are set to −1e9.
        """
        board   = pair_images[0, 0].unsqueeze(0).unsqueeze(0)           # (1, 1, 128, 128)
        deltas  = (pair_images[:, 1] - pair_images[:, 0]).unsqueeze(1)  # (N, 1, 128, 128)
        result  = pair_images[:, 1].unsqueeze(1)                        # (N, 1, 128, 128)
        part_in = torch.cat([deltas, result], dim=1)                    # (N, 2, 128, 128)

        board_feat = self.board_cnn(board)                              # (1, 128)
        part_feats = self.part_cnn(part_in)                             # (N, 128)

        fused = self.fusion(torch.cat([
            board_feat.expand(part_feats.shape[0], -1),
            part_feats,
        ], dim=-1))                                                      # (N, 128)

        ctx    = self.part_ctx(fused, mask)                             # (N, 128)
        logits = self.part_head(ctx).squeeze(-1)                        # (N,)
        return logits.masked_fill(mask == 0, -1e9)


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
    N_PARTS = 6
    device  = torch.device("cpu")

    for name, model in [("SplitPointerNet", SplitPointerNet()),
                         ("SpatialPointerNet", SpatialPointerNet())]:
        model.eval()
        pair_images = torch.rand(N_PARTS, 2, 128, 128)
        mask        = torch.ones(N_PARTS)
        mask[2]     = 0.0  # mark part 2 as placed

        logits = model(pair_images, mask)
        assert logits.shape == (N_PARTS,), f"expected ({N_PARTS},), got {tuple(logits.shape)}"
        assert logits[2].item() == -1e9, "masked part should be -1e9"
        print(f"{name} shape check OK — logits: {tuple(logits.shape)}")

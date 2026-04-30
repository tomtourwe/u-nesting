"""
Supervised validation: can a ValueNet predict rollout_value() from board SDF + scalars?

Architecture: small CNN on board SDF (1, 128, 128) + two scalars
  [n_remaining / N, remaining_area / board_area]

Key insight: absolute prediction accuracy doesn't matter — PPO normalises advantages.
What matters is that V(s_{t+1}) - V(s_t) has the right sign and relative ordering.
Primary metric: Spearman correlation of predicted differences vs actual differences.

Optimised for Apple M3:
  - Parallel data collection across CPU cores (multiprocessing)
  - MPS backend for training
  - Batched evaluation (no per-sample forward passes)

Usage:
    maturin develop --release --features python-bindings
    python scripts/train_value_net.py \\
        --json data/aligner_svgs.json \\
        --plate-width 280 --plate-height 200 \\
        --n-parts 15 --collect-episodes 300 --epochs 100
"""

import argparse
import math
import multiprocessing
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from u_nesting_gym import UNestingGymEnv, _rasterize_polygon

import wandb


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# ValueNet — board SDF + scalars only
# ---------------------------------------------------------------------------

class ValueNet(nn.Module):
    """
    Predicts rollout_value() from board geometry + remaining parts statistics.

    Input:
        sdf     : (B, 1, 128, 128) — current board SDF (free space geometry)
        scalars : (B, 2)           — [n_remaining/N, remaining_area/board_area]

    The CNN captures whether free space is large+contiguous or fragmented.
    The scalars capture how much material still needs to fit.
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(1024, 128), nn.ReLU(True),
        )
        self.scalar_proj = nn.Sequential(nn.Linear(2, 32), nn.ReLU(True))
        self.value_head  = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(True),
            nn.Linear(64, 1),
        )

    def forward(self, sdf: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.value_head(
            torch.cat([self.cnn(sdf), self.scalar_proj(scalars)], dim=-1)
        ).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polygon_area(polygon: list) -> float:
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j     = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _board_sdf(env: UNestingGymEnv) -> np.ndarray:
    """Current board SDF — (128, 128) float32, one EDT call."""
    IMG    = env.IMG_SIZE
    canvas = np.zeros((IMG, IMG), dtype=bool)
    for verts in env.placed_polygons():
        verts_px = [(x * env._sx, y * env._sy) for x, y in verts]
        canvas  |= _rasterize_polygon(verts_px, IMG)
    if canvas.any():
        dist = distance_transform_edt(~canvas).astype(np.float32)
    else:
        dist = np.full((IMG, IMG), np.inf, dtype=np.float32)
    return np.clip(dist, 0, env._sdf_clip_px) / env._sdf_clip_px


def _scalars(env: UNestingGymEnv, n_parts: int, board_area: float) -> np.ndarray:
    remaining     = env.remaining_item_ids()
    episode_geoms = env._episode_geoms()
    rem_area      = sum(_polygon_area(episode_geoms[i]["polygon"]) for i in remaining)
    return np.array([len(remaining) / n_parts, rem_area / board_area], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data collection (single worker — called in subprocess)
# ---------------------------------------------------------------------------

def _collect_worker(worker_args: tuple) -> list:
    """
    Collects episodes in a subprocess. Each subprocess creates its own env
    so the NFP cache warms up independently per worker.

    Returns list of episodes; each episode is a list of step dicts.
    """
    json_path, plate_w, plate_h, rotations_arg, n_parts, n_episodes, seed = worker_args

    env = UNestingGymEnv(
        json_path=json_path,
        plate_width=plate_w,
        plate_height=plate_h,
        rotations=rotations_arg,
    )
    board_area = plate_w * plate_h
    rng        = random.Random(seed)
    episodes   = []

    for _ in range(n_episodes):
        lib_ids = env.sample_episode_ids(n_parts, rng)
        # Largest-first ordering — matches rollout_value() internals
        order   = sorted(range(n_parts),
                         key=lambda i: _polygon_area(env._library[lib_ids[i]]["polygon"]),
                         reverse=True)
        env.reset(lib_ids)

        steps = []
        for episode_id in order:
            remaining = env.remaining_item_ids()
            if not remaining:
                break
            steps.append({
                "sdf":     _board_sdf(env).astype(np.float16),
                "scalars": _scalars(env, n_parts, board_area),
                "value":   float(env.rollout_value()[0]),
            })
            if episode_id in remaining:
                env.place_anywhere(episode_id)

        if steps:
            episodes.append(steps)

    return episodes


def collect_parallel(args: argparse.Namespace) -> list:
    """Distribute episode collection across CPU cores."""
    n_workers    = min(args.workers, args.collect_episodes)
    eps_per_worker = [args.collect_episodes // n_workers] * n_workers
    # distribute remainder
    for i in range(args.collect_episodes % n_workers):
        eps_per_worker[i] += 1

    rotations_arg = [int(r) for r in args.rotations] if args.rotations else None
    worker_args   = [
        (args.json, args.plate_width, args.plate_height,
         rotations_arg, args.n_parts, eps_per_worker[i], args.seed + i)
        for i in range(n_workers)
    ]

    print(f"  Collecting {args.collect_episodes} episodes across {n_workers} workers…")
    t0 = time.time()

    # Use spawn context — required on macOS for subprocesses with Rust extensions
    ctx  = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_collect_worker, worker_args)

    episodes = [ep for worker_result in results for ep in worker_result]
    n_steps  = sum(len(e) for e in episodes)
    vals     = [s["value"] for e in episodes for s in e]
    print(f"  Done in {time.time()-t0:.0f}s — "
          f"{len(episodes)} episodes, {n_steps} steps  "
          f"(value range {min(vals):.3f}–{max(vals):.3f}  mean={np.mean(vals):.3f})")
    return episodes


# ---------------------------------------------------------------------------
# Flatten + consecutive pairs
# ---------------------------------------------------------------------------

def flatten(episodes: list):
    sdfs, scalars_list, values = [], [], []
    for ep in episodes:
        for s in ep:
            sdfs.append(s["sdf"])
            scalars_list.append(s["scalars"])
            values.append(s["value"])
    return sdfs, scalars_list, np.array(values, dtype=np.float32)


def consecutive_pairs(episodes: list):
    """(sdf_t, sc_t, sdf_{t+1}, sc_{t+1}, dV_actual) for each consecutive step pair."""
    pairs = []
    for ep in episodes:
        for i in range(len(ep) - 1):
            pairs.append((
                ep[i]["sdf"],     ep[i]["scalars"],
                ep[i+1]["sdf"], ep[i+1]["scalars"],
                float(ep[i+1]["value"] - ep[i]["value"]),
            ))
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StepDataset(torch.utils.data.Dataset):
    def __init__(self, sdfs, scalars_list, values):
        self.sdfs    = sdfs
        self.scalars = scalars_list
        self.values  = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        sdf = torch.from_numpy(self.sdfs[idx].astype(np.float32)).unsqueeze(0)
        sc  = torch.from_numpy(self.scalars[idx])
        val = torch.tensor(self.values[idx], dtype=torch.float32)
        return sdf, sc, val


# ---------------------------------------------------------------------------
# Batched prediction helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_batch(model: ValueNet, sdfs: list, scalars: list,
                   device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Run batched forward passes over a list of numpy arrays."""
    model.eval()
    all_preds = []
    for start in range(0, len(sdfs), batch_size):
        sdf_b = torch.from_numpy(
            np.stack([s.astype(np.float32) for s in sdfs[start:start+batch_size]])
        ).unsqueeze(1).to(device)                                        # (B, 1, 128, 128)
        sc_b  = torch.from_numpy(
            np.stack(scalars[start:start+batch_size])
        ).to(device)                                                     # (B, 2)
        all_preds.append(model(sdf_b, sc_b).cpu().numpy())
    return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: ValueNet,
             sdfs: list, scalars: list, values: np.ndarray,
             pairs: list,
             device: torch.device):
    preds = _predict_batch(model, sdfs, scalars, device)
    mae   = float(np.abs(preds - values).mean())

    abs_corr, _ = spearmanr(preds, values)
    abs_corr     = 0.0 if math.isnan(abs_corr) else float(abs_corr)

    dv_corr = sign_acc = 0.0
    if pairs:
        sdfs_t  = [p[0] for p in pairs]
        scs_t   = [p[1] for p in pairs]
        sdfs_t1 = [p[2] for p in pairs]
        scs_t1  = [p[3] for p in pairs]
        dv_act  = np.array([p[4] for p in pairs], dtype=np.float32)

        v_t  = _predict_batch(model, sdfs_t,  scs_t,  device)
        v_t1 = _predict_batch(model, sdfs_t1, scs_t1, device)
        dv_pred = v_t1 - v_t

        r, _ = spearmanr(dv_pred, dv_act)
        dv_corr  = 0.0 if math.isnan(r) else float(r)
        sign_acc = float(np.mean(np.sign(dv_pred) == np.sign(dv_act))) * 100

    return mae, abs_corr, dv_corr, sign_acc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model: ValueNet, loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total, n = 0.0, 0
    for sdf_b, sc_b, val_b in loader:
        pred = model(sdf_b.to(device), sc_b.to(device))
        loss = F.mse_loss(pred, val_b.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * sdf_b.shape[0]
        n     += sdf_b.shape[0]
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",             required=True)
    parser.add_argument("--plate-width",      type=float, required=True)
    parser.add_argument("--plate-height",     type=float, required=True)
    parser.add_argument("--n-parts",          type=int,   default=15)
    parser.add_argument("--rotations",        type=float, nargs="+", default=None)
    parser.add_argument("--collect-episodes", type=int,   default=300)
    parser.add_argument("--workers",          type=int,   default=6,
                        help="parallel workers for data collection (default 6 for M3)")
    parser.add_argument("--epochs",           type=int,   default=100)
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--out",              type=str,   default="value_net.pt")
    parser.add_argument("--wandb-project",    type=str,   default="u-nesting-value-net")
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = _device()
    print(f"Device: {device}")

    # ── Collect (parallel across CPU cores) ────────────────────────────────
    episodes = collect_parallel(args)

    # ── Split by episode ───────────────────────────────────────────────────
    idx    = np.random.permutation(len(episodes))
    split  = int(0.8 * len(episodes))
    tr_eps = [episodes[i] for i in idx[:split]]
    va_eps = [episodes[i] for i in idx[split:]]

    tr_sdfs,  tr_sc,  tr_vals  = flatten(tr_eps)
    va_sdfs,  va_sc,  va_vals  = flatten(va_eps)
    va_pairs                   = consecutive_pairs(va_eps)

    tr_loader = torch.utils.data.DataLoader(
        StepDataset(tr_sdfs, tr_sc, tr_vals),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False,
    )
    print(f"Train: {len(tr_vals)} steps ({len(tr_eps)} eps)  "
          f"Val: {len(va_vals)} steps ({len(va_eps)} eps)  "
          f"Val pairs: {len(va_pairs)}")

    # ── Model ──────────────────────────────────────────────────────────────
    model     = ValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"ValueNet parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    wandb.init(project=args.wandb_project, config=vars(args))

    # ── Train ──────────────────────────────────────────────────────────────
    best_dv_corr = -1.0

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, tr_loader, optimizer, device)

        tr_mae, tr_corr, _, _                 = evaluate(model, tr_sdfs, tr_sc, tr_vals, [], device)
        va_mae, va_corr, dv_corr, sign_acc    = evaluate(model, va_sdfs, va_sc, va_vals, va_pairs, device)

        print(
            f"Epoch {epoch:>4}/{args.epochs}  "
            f"train_mae={tr_mae:.4f}  "
            f"val_mae={va_mae:.4f}  "
            f"abs_corr={va_corr:.3f}  "
            f"dv_corr={dv_corr:.3f}  "
            f"sign_acc={sign_acc:.1f}%"
        )
        wandb.log({
            "train/mae":    tr_mae,
            "val/mae":      va_mae,
            "val/abs_corr": va_corr,
            "val/dv_corr":  dv_corr,
            "val/sign_acc": sign_acc,
            "epoch":        epoch,
        })

        if dv_corr > best_dv_corr:
            best_dv_corr = dv_corr
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "dv_corr": dv_corr, "args": vars(args)}, args.out)
            print(f"  ✓ saved  (best dv_corr={best_dv_corr:.3f})")

    print(f"\nDone. Best dv_corr={best_dv_corr:.3f}  checkpoint: {args.out}")
    if best_dv_corr > 0.7:
        print("RESULT: dv_corr > 0.7 — differences are predictable, useful for PPO.")
    elif best_dv_corr > 0.4:
        print("RESULT: dv_corr 0.4–0.7 — moderate signal, may still reduce advantage variance.")
    else:
        print("RESULT: dv_corr < 0.4 — differences not predictable from board SDF alone.")

    wandb.finish()


if __name__ == "__main__":
    # Required for multiprocessing with spawn on macOS
    multiprocessing.set_start_method("spawn", force=True)
    main()

"""
Nest a sample of parts from a shape-library JSON into a rectangular plate.

Uses the Board2D incremental API (same as the training script / notebook):
place parts one-by-one in the chosen order; the NFP engine finds the best
position for each part.

Usage:
    python nest_aligner.py --json data/aligner_svgs.json --plate-width 400 --plate-height 250 --n all --sort --plot-out nest.png
    python nest_aligner.py --strategy blf                     # faster, lower quality
    python nest_aligner.py --seed 42                          # reproducible sample
    python nest_aligner.py --no-plot                          # skip the plot
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "crates" / "python" / "python"))


def _normalize_polygon(vertices: list) -> list:
    """Center and rotationally normalise so rotation=0° means the same
    visual orientation for every part: tall, concavity facing up."""
    pts = np.array(vertices, dtype=np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(major[1], major[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    pts = (np.array([[cos_a, -sin_a], [sin_a, cos_a]]) @ pts.T).T
    # Ensure tall orientation
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    if w > h:
        pts = pts[:, ::-1].copy()
    # Orient concavity upward: solid base pulls centroid below bbox mid
    bbox_mid_y = (pts[:, 1].max() + pts[:, 1].min()) / 2.0
    if pts[:, 1].mean() > bbox_mid_y:
        pts[:, 1] = -pts[:, 1]
    return pts.tolist()

_DEFAULT_DATASET = Path(__file__).parent.parent / "repos/sparrow/data/input/aligner_library.json"


def load_sample(n: int | str, seed: int | None, dataset: Path, rotations_override: list[float] | None = None) -> list[dict]:
    with open(dataset) as f:
        data = json.load(f)

    items = data["items"]
    if n == "all":
        sampled = items
    else:
        rng = random.Random(seed)
        sampled = rng.sample(items, int(n))

    return [
        {
            "id": item["label"],
            "polygon": _normalize_polygon(item["shape"]["data"]),
            "rotations": rotations_override if rotations_override is not None else item["allowed_orientations"],
        }
        for item in sampled
    ]


def polygon_area(vertices: list) -> float:
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def run_board2d(geometries: list[dict], strategy: str, boundary: dict,
                rollout_multiplier: float = 1.0, greedy: bool = False) -> tuple:
    """
    Incremental greedy placement via Board2D (same API as the RL training loop).

    If greedy=True, uses place_remaining() — the same coarse greedy used during
    training eval — with rollout_multiplier controlling the sample step.
    If greedy=False, places parts one-by-one with the fine grid (committed quality).

    Returns (snapshots, placed_polygons, placements, utilization).
    """
    from u_nesting import Board2D

    board = Board2D(
        boundary=boundary,
        geometries=geometries,
        config={"strategy": strategy, "spacing": 0.0,
                "rollout_step_multiplier": rollout_multiplier},
    )

    # In greedy mode sort largest-first (mirrors Rust place_remaining order),
    # then place one-by-one to capture per-step snapshots.
    if greedy:
        geometries = sorted(geometries, key=lambda g: polygon_area(g["polygon"]), reverse=True)

    snapshots = []
    placed = 0
    failed = []
    for i, g in enumerate(geometries):
        result = board.place(g["id"])
        if result is not None:
            placed += 1
            util_now = board.packing_density()
            pos = result["position"]
            rot = math.degrees(result["rotation"])
            mode = "greedy" if greedy else "fine"
            print(f"  [{i+1:3d}/{len(geometries)}] placed {g['id']:8s}  pos=({pos[0]:7.1f}, {pos[1]:7.1f})  rot={rot:5.1f}°  density={util_now:.1%}  [{mode}]")
            snapshots.append((
                list(board.placed_polygons()),
                list(board.placements()),
                util_now,
                f"step {placed}\n{g['id']}  {util_now:.0%}",
            ))
        else:
            failed.append(g["id"])
            print(f"  [{i+1:3d}/{len(geometries)}] FAILED {g['id']}")

    util = board.packing_density()
    print(f"\nResult  ({'greedy' if greedy else 'fine'}, multiplier={rollout_multiplier})")
    print(f"  Parts submitted : {len(geometries)}")
    print(f"  Placed          : {placed}")
    print(f"  Failed          : {len(failed)}  {failed}")
    print(f"  Density         : {util:.1%}")

    return snapshots, board.placed_polygons(), board.placements(), util


def _draw_board_state(ax, placed_polygons, placements, boundary, title, cmap):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    w, h = boundary["width"], boundary["height"]
    ax.add_patch(plt.Rectangle((0, 0), w, h, linewidth=1, edgecolor="black", facecolor="#f5f5f5"))
    color_map = {p["geometry_id"]: cmap(i % 20) for i, p in enumerate(placements)}
    for verts, placement in zip(placed_polygons, placements):
        color = color_map[placement["geometry_id"]]
        ax.add_patch(MplPolygon(verts, closed=True, facecolor=(*color[:3], 0.6),
                                edgecolor="black", linewidth=0.3))
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=6)


def plot_steps(
    snapshots: list,
    final_polygons: list,
    final_placements: list,
    boundary: dict,
    save_path: Path | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed")
        return

    cmap = plt.get_cmap("tab20")
    w, h = boundary["width"], boundary["height"]

    # Include final state as last panel
    all_frames = snapshots + [(final_polygons, final_placements, snapshots[-1][2] if snapshots else 0,
                               f"final\n{len(final_placements)} placed")]

    n = len(all_frames)
    ncols = math.ceil(math.sqrt(n * w / h))
    nrows = math.ceil(n / ncols)

    import numpy as np
    cell_w = 3 * w / max(w, h)
    cell_h = 3 * h / max(w, h)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_w, nrows * cell_h))
    axes = np.array(axes).flatten()

    for i, (polys, pls, _, label) in enumerate(all_frames):
        _draw_board_state(axes[i], polys, pls, boundary, label, cmap)

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle(f"Nesting steps — {len(final_placements)} placed, {boundary['width']:.0f}×{boundary['height']:.0f} mm", fontsize=10)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=_DEFAULT_DATASET, help="Shape library JSON")
    parser.add_argument("--plate-width",  type=float, default=100.0)
    parser.add_argument("--plate-height", type=float, default=100.0)
    parser.add_argument("--n", default="25", help="Number of parts to sample, or 'all'")
    parser.add_argument("--strategy", default="nfp", choices=["nfp", "blf", "ga", "brkga", "sa"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rotations", type=float, nargs="+", default=None,
                        help="Override rotations in degrees, e.g. --rotations 0 90 180 270")
    parser.add_argument("--sort", action="store_true", help="Largest-area-first ordering")
    parser.add_argument("--greedy", action="store_true", help="Use place_remaining() — same coarse greedy as training eval")
    parser.add_argument("--rollout-multiplier", type=float, default=3.0, help="Sample step multiplier for greedy (default 3.0, same as training)")
    parser.add_argument("--plot-out", type=Path, default=Path("nest_result.png"), help="Output PNG path (default: nest_result.png)")
    args = parser.parse_args()

    boundary = {"width": args.plate_width, "height": args.plate_height}
    n_label = args.n if args.n == "all" else int(args.n)

    print(f"Loading {n_label} parts from {args.json}, strategy={args.strategy} ...")
    geometries = load_sample(n_label, args.seed, dataset=args.json, rotations_override=args.rotations)

    if args.sort:
        geometries.sort(key=lambda g: polygon_area(g["polygon"]), reverse=True)
        print(f"  Sorted largest-first (areas {polygon_area(geometries[0]['polygon']):.0f} → {polygon_area(geometries[-1]['polygon']):.0f} mm²)")

    snapshots, placed_polys, placements, _ = run_board2d(
        geometries, args.strategy, boundary,
        rollout_multiplier=args.rollout_multiplier,
        greedy=args.greedy,
    )

    plot_steps(snapshots, placed_polys, placements, boundary, save_path=args.plot_out)


if __name__ == "__main__":
    main()

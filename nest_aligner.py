"""
Nest a random sample of parts from aligner_library.json into a 100x100 square.

Usage:
    python nest_aligner.py                   # uses NFP strategy, 25 random parts
    python nest_aligner.py --n 30            # 30 random parts
    python nest_aligner.py --strategy blf    # faster, lower quality
    python nest_aligner.py --seed 42         # reproducible sample
    python nest_aligner.py --no-plot         # skip the plot
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

DATASET = Path(__file__).parent.parent / "repos/sparrow/data/input/aligner_library.json"
BOUNDARY = {"width": 100.0, "height": 100.0}


def load_sample(n: int, seed: int | None, rotations_override: list[float] | None = None) -> list[dict]:
    with open(DATASET) as f:
        data = json.load(f)

    rng = random.Random(seed)
    items = rng.sample(data["items"], n)

    geometries = []
    for item in items:
        geometries.append({
            "id": item["label"],
            "polygon": item["shape"]["data"],
            "quantity": 1,
            "rotations": rotations_override if rotations_override is not None else item["allowed_orientations"],
        })
    return geometries


def run_via_json(geometries: list[dict], strategy: str, time_limit_ms: int) -> dict:
    """
    Invoke the nester via the JSON API (FFI / CLI path).
    Falls back to printing the request JSON if the native module isn't built yet.
    """
    try:
        import u_nesting  # built with maturin
        result = u_nesting.solve_2d(
            geometries=geometries,
            boundary=BOUNDARY,
            config={
                "strategy": strategy,
                "spacing": 0.0,
                "time_limit_ms": time_limit_ms,
            },
        )
        return result
    except ModuleNotFoundError:
        # Module not built yet — dump the request JSON so you can test another way
        request = {
            "geometries": geometries,
            "boundary": BOUNDARY,
            "config": {
                "strategy": strategy,
                "spacing": 0.0,
                "time_limit_ms": time_limit_ms,
            },
        }
        out = Path("aligner_request.json")
        out.write_text(json.dumps(request, indent=2))
        print(f"[!] u_nesting module not found — request written to {out}")
        print("    Build the Python bindings first (see instructions below).")
        sys.exit(1)


def print_result(result: dict, n_parts: int) -> None:
    placed = len(result["placements"])
    unplaced = len(result["unplaced"])
    util = result["utilization"] * 100
    ms = result["computation_time_ms"]

    print(f"\nResult")
    print(f"  Parts sampled : {n_parts}")
    print(f"  Placed        : {placed}")
    print(f"  Unplaced      : {unplaced}  {result['unplaced']}")
    print(f"  Utilization   : {util:.1f}%")
    print(f"  Time          : {ms / 1000:.2f}s")

    if result.get("error"):
        print(f"  Error         : {result['error']}")


def rotate(polygon: list, angle_rad: float) -> list:
    """Rotate polygon vertices around the origin by angle in radians."""
    if angle_rad == 0.0:
        return polygon
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return [
        [x * cos_a - y * sin_a, x * sin_a + y * cos_a]
        for x, y in polygon
    ]


def translate(polygon: list, tx: float, ty: float) -> list:
    return [[x + tx, y + ty] for x, y in polygon]


def plot_result(
    result: dict,
    geometries: list[dict],
    boundary: dict,
    save_path: Path | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
    except ImportError:
        print("[!] matplotlib not installed — run: pip install matplotlib")
        return

    # Build lookup: id -> polygon vertices
    geom_map = {g["id"]: g["polygon"] for g in geometries}

    w = boundary["width"]
    h = boundary["height"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw boundary
    ax.add_patch(plt.Rectangle((0, 0), w, h, linewidth=2, edgecolor="black", facecolor="#f5f5f5"))

    # Color cycle
    cmap = plt.get_cmap("tab20")
    placed_ids = [p["geometry_id"] for p in result["placements"]]
    color_map = {gid: cmap(i % 20) for i, gid in enumerate(sorted(set(placed_ids)))}

    for placement in result["placements"]:
        gid = placement["geometry_id"]
        poly = geom_map.get(gid)
        if poly is None:
            continue

        angle = placement["rotation"][0] if placement["rotation"] else 0.0
        tx, ty = placement["position"][0], placement["position"][1]

        transformed = translate(rotate(poly, angle), tx, ty)

        color = color_map[gid]
        patch = MplPolygon(transformed, closed=True, facecolor=(*color[:3], 0.6), edgecolor="black", linewidth=0.8)
        ax.add_patch(patch)


    # Mark unplaced
    unplaced = result.get("unplaced", [])

    util = result["utilization"] * 100
    title = f"Nesting result — {len(result['placements'])} placed, {len(unplaced)} unplaced — {util:.1f}% utilization"
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-2, w + 2)
    ax.set_ylim(-2, h + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved to {save_path}")
    else:
        plt.show()


def polygon_area(vertices: list) -> float:
    """Shoelace formula for polygon area."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="Number of parts to sample")
    parser.add_argument("--strategy", default="nfp", choices=["nfp", "blf", "ga", "brkga", "sa"])
    parser.add_argument("--time-limit", type=int, default=30000, help="Time limit in ms")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save", action="store_true", help="Save result to result.json")
    parser.add_argument("--rotations", type=float, nargs="+", default=None,
                        help="Override allowed rotations in degrees, e.g. --rotations 0 90 180 270")
    parser.add_argument("--sort", action="store_true", help="Sort parts largest area first before nesting")
    parser.add_argument("--no-plot", action="store_true", help="Skip the plot")
    parser.add_argument("--plot-out", type=Path, default=None, help="Save plot to file instead of showing it (e.g. nest.png)")
    args = parser.parse_args()

    print(f"Sampling {args.n} parts (seed={args.seed}), strategy={args.strategy} ...")
    geometries = load_sample(args.n, args.seed, rotations_override=args.rotations)

    if args.sort:
        geometries.sort(key=lambda g: polygon_area(g["polygon"]), reverse=True)

    result = run_via_json(geometries, args.strategy, args.time_limit)
    print_result(result, args.n)

    if args.save:
        Path("result.json").write_text(json.dumps(result, indent=2))
        print("\n  Result saved to result.json")

    if not args.no_plot:
        plot_result(result, geometries, BOUNDARY, save_path=args.plot_out)


if __name__ == "__main__":
    main()

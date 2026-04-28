"""
Extract aligner polygon shapes from SVG nesting files and write a shape-library
JSON compatible with u-nesting (UNestingGymEnv / nesting_playground notebook).

SVG units are 0.1 mm, so coordinates are divided by 10 to give millimetres.
Each polygon is normalised to have its bounding-box min corner at (0, 0).

Usage:
    python scripts/extract_svg_library.py          # writes data/aligner_svgs.json
    python scripts/extract_svg_library.py --help
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_build_plate(svg_text: str) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) of the white build-plate rectangle."""
    m = re.search(
        r'id="build_plate"[^>]*>.*?d="([^"]+)"', svg_text, re.DOTALL
    )
    if not m:
        raise ValueError("build_plate group not found")
    coords = re.findall(r"[-\d.]+,[-\d.]+", m.group(1))
    pts = [tuple(float(v) for v in c.split(",")) for c in coords]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _parse_path_d(d: str) -> list[tuple[float, float]]:
    """
    Parse a simple SVG path of the form  'M x,y x,y ... z'
    (only M implicit-lineto sequences, no curves or L commands).
    """
    # Strip leading 'M' and trailing 'z'/'Z'
    d = re.sub(r"[Mm]", "", d).strip()
    d = re.sub(r"[Zz]", "", d).strip()
    tokens = re.findall(r"[-\d.]+,[-\d.]+", d)
    return [tuple(float(v) for v in t.split(",")) for t in tokens]


def extract_parts(svg_path: Path) -> tuple[list[list[tuple[float, float]]], float, float]:
    """
    Return (polygons, plate_width_mm, plate_height_mm).

    Each polygon is in millimetres with its own bounding-box min at (0, 0).
    """
    text = svg_path.read_text()

    bx0, by0, bx1, by1 = _parse_build_plate(text)
    scale = 0.1  # SVG units are 0.1 mm → mm

    parts = []
    for m in re.finditer(
        r'id="part-\d+"[^>]*>.*?d="([^"]+)"', text, re.DOTALL
    ):
        raw_verts = _parse_path_d(m.group(1))
        xs = [v[0] for v in raw_verts]
        ys = [v[1] for v in raw_verts]
        ox, oy = min(xs), min(ys)
        poly = [
            (round((x - ox) * scale, 4), round((y - oy) * scale, 4))
            for x, y in raw_verts
        ]
        parts.append(poly)

    return parts, (bx1 - bx0) * scale, (by1 - by0) * scale


# ── deduplication ─────────────────────────────────────────────────────────────

def _normalise(poly: list[tuple[float, float]]) -> tuple:
    """Translate polygon so its bounding-box min corner is at (0, 0)."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    ox, oy = min(xs), min(ys)
    norm = tuple(
        (round(x - ox, 2), round(y - oy, 2)) for x, y in poly
    )
    return norm


def _shape_key(poly: list[tuple[float, float]]) -> str:
    """A hashable string key for a normalised polygon (vertex-order aware)."""
    norm = _normalise(poly)
    # Try all rotations of the vertex list; use the lexicographically smallest
    # as the canonical form so that polygons differing only in start vertex
    # are treated as identical.
    n = len(norm)
    rotations = (norm[i:] + norm[:i] for i in range(n))
    canonical = min(rotations)
    return str(canonical)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg-dir",
        default="data/svgs",
        help="Directory containing the *.svg files (default: data/svgs)",
    )
    parser.add_argument(
        "--out",
        default="data/aligner_svgs.json",
        help="Output JSON path (default: data/aligner_svgs.json)",
    )
    args = parser.parse_args()

    svg_dir = Path(args.svg_dir)
    svg_files = sorted(svg_dir.glob("*.svg"))
    if not svg_files:
        sys.exit(f"No SVG files found in {svg_dir}")

    seen: dict[str, list] = {}   # key → polygon (first occurrence wins)
    plate_dims: dict[str, tuple[float, float]] = {}   # svg_stem → (w, h)

    for svg_file in svg_files:
        print(f"Processing {svg_file.name} …", end=" ", flush=True)
        polys, pw, ph = extract_parts(svg_file)
        plate_dims[svg_file.stem] = (pw, ph)
        before = len(seen)
        for poly in polys:
            key = _shape_key(poly)
            if key not in seen:
                seen[key] = poly
        print(f"{len(polys)} parts → {len(seen) - before} new unique shapes")

    print(f"\nTotal unique shapes: {len(seen)}")

    # Build library items (polygons already normalised to origin in extract_parts)
    items = []
    for i, (key, poly) in enumerate(seen.items()):
        items.append(
            {
                "id": i,
                "label": f"AZ{i:03d}",
                "demand": 1,
                "allowed_orientations": [0.0],
                "shape": {
                    "type": "simple_polygon",
                    "data": [[x, y] for x, y in poly],
                },
            }
        )

    # Determine strip_height from the most common plate height
    heights = [h for (_, h) in plate_dims.values()]
    strip_height = round(max(heights), 4)

    library = {
        "name": "aligner_svgs",
        "strip_height": strip_height,
        "plate_dims": {
            stem: {"width": round(w, 4), "height": round(h, 4)}
            for stem, (w, h) in plate_dims.items()
        },
        "items": items,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(library, indent=2))

    print(f"\nWrote {len(items)} shapes → {out_path}")
    print("\nPlate dimensions (mm):")
    for stem, (w, h) in plate_dims.items():
        print(f"  {stem}: {w:.1f} × {h:.1f} mm  (strip_height={h:.1f})")
    print(f"\nstrip_height in JSON: {strip_height}")
    print("\nExample gym usage:")
    example_stem = next(iter(plate_dims))
    ew, eh = plate_dims[example_stem]
    print(f"  UNestingGymEnv('{out_path}', plate_width={ew:.1f}, plate_height={eh:.1f})")


if __name__ == "__main__":
    main()

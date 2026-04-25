"""
U-Nesting: 2D/3D Spatial Optimization Engine

A high-performance library for 2D polygon nesting and 3D bin packing.
"""

import math

import numpy as np

from .u_nesting import solve_2d, solve_3d, version, available_strategies
from .u_nesting import Board2D as _Board2D  # Rust class


def _rasterize_polygon(vertices_px: list, size: int) -> np.ndarray:
    """Fill a polygon (pixel coordinates) onto a boolean canvas."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    pts = [(float(x), float(y)) for x, y in vertices_px]
    if len(pts) >= 3:
        draw.polygon(pts, fill=255)
    return np.array(img, dtype=np.uint8) > 0


def _sdf_from_mask(mask: np.ndarray, sdf_clip_px: int) -> np.ndarray:
    """Truncated unsigned SDF: 0=inside part, 1=≥sdf_clip_px away."""
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~mask)
    return np.clip(dist, 0, sdf_clip_px).astype(np.float32) / sdf_clip_px


class Board2D:
    """
    Stateful 2D nesting board for incremental placement.

    Wraps the Rust engine and adds image rendering for RL observations.

    Args:
        boundary   : dict with ``width``/``height`` (or ``polygon``)
        geometries : list of geometry dicts — ``{"id", "polygon", "rotations", ...}``
        config     : optional config dict — ``{"strategy", "spacing", ...}``
        img_size   : pixel size of rendered images (default 128)
        sdf_clip_px: SDF truncation distance in pixels (default 8)

    Example::

        board = Board2D(
            boundary={"width": 100.0, "height": 100.0},
            geometries=parts,
        )
        result = board.place("A000")   # None if doesn't fit
        img    = board.render_placement(result)   # (128, 128) float32 SDF
        full   = board.board_image()              # (1, 128, 128) float32 SDF
    """

    IMG_SIZE = 128

    def __init__(
        self,
        boundary: dict,
        geometries: list,
        config: dict | None = None,
        img_size: int = 128,
        sdf_clip_px: int = 8,
    ):
        self._board = _Board2D(boundary=boundary, geometries=geometries, config=config)
        self._geom_verts: dict[str, list] = {g["id"]: g["polygon"] for g in geometries}
        self._plate_w: float = boundary.get("width", 1.0)
        self._plate_h: float = boundary.get("height", 1.0)
        self._img_size: int = img_size
        self._sdf_clip_px: int = sdf_clip_px
        self._sx: float = img_size / self._plate_w
        self._sy: float = img_size / self._plate_h

    # ── Core placement API (delegates to Rust) ────────────────────────────────

    def place(self, geometry_id: str) -> dict | None:
        """
        Place the named part at the best valid position.

        Returns ``{"geometry_id", "position": [x, y], "rotation": r}`` or
        ``None`` if the part doesn't fit.
        """
        return self._board.place(geometry_id)

    def undo(self) -> bool:
        """Remove the last placed part. Returns True if something was removed."""
        return self._board.undo()

    def reset(self) -> None:
        """Remove all placed parts."""
        self._board.reset()

    def start_episode(self, ids: list) -> None:
        """
        Start a new episode using a subset of the board's library.

        Clears placed parts and sets the active episode to ``ids`` (list of
        geometry ID strings).  The NFP cache is preserved so cached computations
        from previous episodes are reused, making training faster over time.
        """
        self._board.start_episode(ids)

    def snapshot(self) -> list:
        """Lightweight board snapshot. Pass to ``restore()`` to rewind."""
        return self._board.snapshot()

    def restore(self, snapshot: list) -> None:
        """Restore board to a previously snapshotted state."""
        self._board.restore(snapshot)

    def utilization(self) -> float:
        """Current fill ratio (0.0–1.0)."""
        return self._board.utilization()

    def placed_count(self) -> int:
        """Number of parts currently on the board."""
        return self._board.placed_count()

    def placements(self) -> list:
        """Current placements as a list of dicts."""
        return self._board.placements()

    def placed_polygons(self) -> list:
        """Board-space polygon vertices for every placed part."""
        return self._board.placed_polygons()

    def n_placed(self) -> int:
        """Number of parts currently placed."""
        return self._board.n_placed()

    def remaining_ids(self) -> list:
        """Ordered list of geometry IDs not yet placed this episode."""
        return self._board.remaining_ids()

    def lbf_preview_all(self, ids: list) -> list:
        """Speculatively place each id via LBF; return board-space vertices or None."""
        return self._board.lbf_preview_all(ids)

    def lbf_rollout_value(self) -> tuple:
        """Snapshot → greedy LBF fill → restore. Returns (packing_density, n_placed)."""
        return self._board.lbf_rollout_value()

    def lbf_place_all(self) -> int:
        """Place all remaining parts via LBF. Returns count placed."""
        return self._board.lbf_place_all()

    def packing_density(self) -> float:
        """placed_area / bbox_area (0.0 if empty)."""
        return self._board.packing_density()

    def bbox_area(self) -> float:
        """Axis-aligned bounding box area of all placed parts."""
        return self._board.bbox_area()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def board_image(self) -> np.ndarray:
        """
        Render all placed parts as a ``(1, img_size, img_size)`` float32 SDF.

        Convention (matches sparrow): 0.0 = occupied, 1.0 = empty / far away.
        Returns all-ones on an empty board.
        """
        canvas = np.zeros((self._img_size, self._img_size), dtype=bool)
        for verts_board in self._board.placed_polygons():
            verts_px = [(x * self._sx, y * self._sy) for x, y in verts_board]
            canvas |= _rasterize_polygon(verts_px, self._img_size)

        if not canvas.any():
            return np.ones((1, self._img_size, self._img_size), dtype=np.float32)

        return _sdf_from_mask(canvas, self._sdf_clip_px)[np.newaxis]

    def render_placement(self, placement: dict) -> np.ndarray:
        """
        Render a placed part centered on a canvas as an
        ``(img_size, img_size)`` float32 SDF, with rotation applied.

        Like ``part_canvas`` but with the placement rotation, so the
        orientation is clearly visible regardless of board position.

        Args:
            placement: dict returned by ``place()`` —
                       ``{"geometry_id", "position": [x, y], "rotation": r}``

        Returns:
            ``(img_size, img_size)`` float32, 0=inside part, 1=far away.
        """
        gid = placement["geometry_id"]
        angle_rad = placement["rotation"]

        raw = self._geom_verts.get(gid)
        if raw is None:
            return np.ones((self._img_size, self._img_size), dtype=np.float32)

        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        verts_rot = [
            (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
            for x, y in raw
        ]

        pts = np.array(verts_rot, dtype=np.float64)
        cx_board = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
        cy_board = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
        cx_px = self._img_size / 2.0
        cy_px = self._img_size / 2.0
        verts_px = [
            (v[0] * self._sx - cx_board * self._sx + cx_px,
             v[1] * self._sy - cy_board * self._sy + cy_px)
            for v in verts_rot
        ]
        mask = _rasterize_polygon(verts_px, self._img_size)
        return _sdf_from_mask(mask, self._sdf_clip_px)

    def part_canvas(self, geometry_id: str) -> np.ndarray:
        """
        Render a part centered on a blank canvas as an
        ``(img_size, img_size)`` float32 SDF.

        Useful as a static per-part feature for the RL agent — the image
        doesn't change during the episode.
        """
        raw = self._geom_verts.get(geometry_id)
        if raw is None:
            return np.ones((self._img_size, self._img_size), dtype=np.float32)

        pts = np.array(raw, dtype=np.float64)
        cx_board = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
        cy_board = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
        cx_px = self._img_size / 2.0
        cy_px = self._img_size / 2.0
        verts_px = [
            (v[0] * self._sx - cx_board * self._sx + cx_px,
             v[1] * self._sy - cy_board * self._sy + cy_px)
            for v in raw
        ]
        mask = _rasterize_polygon(verts_px, self._img_size)
        return _sdf_from_mask(mask, self._sdf_clip_px)


__all__ = [
    "solve_2d", "solve_3d", "version", "available_strategies",
    "Board2D",
]
__version__ = version()

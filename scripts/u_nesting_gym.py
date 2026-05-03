"""
RL Environment — bridges the neural network and the U-Nesting engine.

The agent predicts which part to place next; the engine's NFP-guided heuristic
determines the (x, y, rotation).  The episode ends when no more parts fit.

Library JSON format (u-nesting format):
    [
        {"id": "A000", "polygon": [[x, y], ...], "rotations": [0, 90, 180, 270]},
        ...
    ]

Build the Rust extension before importing:
    maturin develop --release --features python-bindings
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "crates" / "python" / "python"))

from u_nesting import Board2D as _RustBoard2D


def _normalize_polygon(vertices: list) -> list:
    """
    Center and rotationally normalise a polygon so rotation angles are
    visually consistent across all parts.

    Steps:
      1. Translate centroid to origin.
      2. PCA: rotate so the principal axis aligns with the x-axis.
      3. Ensure tall orientation (height ≥ width) — if wider than tall,
         rotate 90° so the long dimension runs vertically.
      4. Resolve 180° flip: the concavity (open end of a U-shape) should
         face upward. The centroid of a U sits toward the solid base, so
         after centering it is displaced downward relative to the bounding
         box mid-point. Flip vertically if the centroid is above the bbox
         mid-point (meaning the solid part is on top, opening faces down).

    Returns a new list of [x, y] pairs (floats).
    """
    pts = np.array(vertices, dtype=np.float64)

    # 1. Centre
    pts -= pts.mean(axis=0)

    # 2. PCA — align principal axis with x-axis
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(major[1], major[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    pts = (np.array([[cos_a, -sin_a], [sin_a, cos_a]]) @ pts.T).T

    # 3. Ensure tall (height ≥ width)
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    if w > h:
        # swap x and y (rotate 90°)
        pts = pts[:, ::-1].copy()

    # 4. Orient concavity upward:
    #    For a U-shape the solid base pulls the centroid down (y < bbox_mid).
    #    If centroid_y > bbox_mid the solid is on top → flip vertically.
    bbox_mid_y = (pts[:, 1].max() + pts[:, 1].min()) / 2.0
    centroid_y = pts[:, 1].mean()
    if centroid_y > bbox_mid_y:
        pts[:, 1] = -pts[:, 1]

    return pts.tolist()


def _rasterize_polygon(vertices_px: list, size: int = 128) -> np.ndarray:
    """Fill a polygon (pixel coordinates) onto a boolean canvas."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    pts = [(float(x), float(y)) for x, y in vertices_px]
    if len(pts) >= 3:
        draw.polygon(pts, fill=255)
    return np.array(img, dtype=np.uint8) > 0


class UNestingGymEnv:
    """
    The RL environment for U-Nesting.

    Each episode, a random set of N parts is selected from a shape library.
    The agent predicts which part to place next; the NFP-guided engine
    determines the (x, y, rotation).

    Args:
        json_path   : path to shape library JSON (u-nesting format)
        plate_width : board width in geometry units
        plate_height: board height in geometry units
        sdf_clip_px : SDF truncation distance in pixels (default 8)
        config      : optional dict forwarded to Board2D (spacing, margin, …)
    """

    IMG_SIZE: int = 128

    def __init__(
        self,
        json_path: str,
        plate_width: float,
        plate_height: float,
        sdf_clip_px: int = 8,
        config: dict | None = None,
        rotations: list[float] | None = None,
    ):
        with open(json_path) as f:
            raw = json.load(f)

        # Accept both a bare list and {"items": [...]} (sparrow-style)
        if isinstance(raw, list):
            self._library: list[dict] = [
                {**g, "polygon": _normalize_polygon(g["polygon"])} for g in raw
            ]
        elif isinstance(raw, dict) and "items" in raw:
            # Convert sparrow format.  Map allowed_orientations → rotations so the
            # Rust engine sees the library's orientation constraints even when no
            # CLI --rotations override is given (greedy baseline, rollout_value, etc.)
            self._library = [
                {
                    "id": item.get("label", str(i)),
                    "polygon": _normalize_polygon(item["shape"]["data"]),
                    "rotations": item.get("allowed_orientations", []),
                }
                for i, item in enumerate(raw["items"])
            ]
        else:
            raise ValueError(f"Unrecognised library format in {json_path}")

        # Rotations to apply to every part (degrees). None = use whatever the
        # library specifies; an explicit list overrides all library rotations.
        self._rotations = rotations

        self.plate_w      = plate_width
        self.plate_h      = plate_height
        self._sdf_clip_px = sdf_clip_px
        self._config      = config

        self._sx = self.IMG_SIZE / plate_width
        self._sy = self.IMG_SIZE / plate_height

        self._boundary = {"width": plate_width, "height": plate_height}

        # Build full-library geometry list with rotations applied upfront.
        # One Board2D is created here and reused across all episodes so the
        # NFP cache warms up over time instead of being discarded each reset.
        full_geoms = (
            [{**g, "rotations": rotations} for g in self._library]
            if rotations is not None
            else self._library
        )
        self._board = _RustBoard2D(
            boundary=self._boundary,
            geometries=full_geoms,
            config=self._config,
        )

        # Set per episode
        self.current_lib_ids: list[int] = []

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def sample_episode_ids(self, n: int = 15, rng: random.Random | None = None) -> list[int]:
        """
        Draw n random part indices from the library (without replacement).

        Training across many different combinations forces the agent to learn
        general packing principles rather than memorising a single layout.
        """
        if rng is None:
            rng = random
        return rng.sample(range(len(self._library)), n)

    def reset(self, lib_ids: list[int]) -> None:
        """
        Start a fresh episode with the given library indices.

        Reuses the shared Board2D (preserving the NFP cache) and activates
        only the selected parts for this episode. Episode IDs (0, 1, 2, …)
        map to lib_ids positionally.
        """
        self.current_lib_ids = list(lib_ids)
        episode_str_ids = [self._library[i]["id"] for i in lib_ids]
        self._board.start_episode(episode_str_ids)

    # ------------------------------------------------------------------
    # Core step interface
    # ------------------------------------------------------------------

    def place_anywhere(self, episode_id: int) -> bool:
        """
        Let the Rust engine place part `episode_id` at the best valid position.

        Returns True if the part was placed, False if it doesn't fit.
        """
        geom_id = self._episode_geoms()[episode_id]["id"]
        return self._board.place(geom_id) is not None

    def place_with_rotation(self, episode_id: int, rotation_rad: float) -> bool:
        """
        Place part `episode_id` at the agent-specified rotation (radians).

        The NFP engine finds the best (x, y) for that rotation only.
        Returns True if placed, False if the part doesn't fit at that rotation.
        """
        geom_id = self._episode_geoms()[episode_id]["id"]
        return self._board.place_with_rotation(geom_id, rotation_rad) is not None

    def get_rotation_angles(self, n_rotations: int = 8) -> list[float]:
        """Return `n_rotations` evenly-spaced angles in [0, 2π)."""
        import math
        return [2 * math.pi * r / n_rotations for r in range(n_rotations)]

    def preview_images_per_rotation(
        self, rotations_rad: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (N, 2R+1, IMG_SIZE, IMG_SIZE) observation images plus per-(part,rotation)
        scalar features.

        Channel 0        : current board SDF (same for all N parts).
        Channels 1..R+1  : hypothetical board SDF after placing part i at
                           rotation rotations_rad[r].  Falls back to the current
                           SDF if the part is already placed or doesn't fit at
                           that rotation.
        Channels R+1..2R+1: isolated part SDF at each rotation (part only,
                           centered on canvas, no board context). Lets the
                           network distinguish part shapes independently of
                           placement position.  Falls back to all-ones (no-part
                           signal) when the part doesn't fit at that rotation.

        Scalar features (N, R, 3) — z-scored across N parts per rotation per feature
        so that even small size differences become discriminatory:
            feat[0] = part pixel area / (IMG*IMG)   (relative footprint)
            feat[1] = bbox_width  / IMG             (normalised width)
            feat[2] = bbox_height / IMG             (normalised height)
        Fallback value for non-fitting (part, rotation) pairs: 0.0 before z-scoring.

        Uses the same min-distance EDT trick as preview_pair_images:
        EDT runs once on the base board and once per (part, rotation) candidate.

        Args:
            rotations_rad: list of R rotation angles in radians.

        Returns:
            images  : np.ndarray of shape (N, 2R+1, IMG_SIZE, IMG_SIZE), dtype float32.
            scalars : np.ndarray of shape (N, R, 3), dtype float32, z-scored across N.
        """
        from scipy.ndimage import distance_transform_edt

        R             = len(rotations_rad)
        N             = len(self._episode_geoms())
        IMG           = self.IMG_SIZE
        half          = IMG / 2.0
        remaining_eps = self.remaining_item_ids()
        episode_geoms = self._episode_geoms()

        # ── base canvas + board SDF ────────────────────────────────────────
        # Include a 1-pixel border so the SDF encodes distance to nearest
        # obstacle (placed part OR plate wall).  Without this, near-wall gaps
        # look identical to open-centre gaps (both clip to 1.0), hiding the
        # wasted edge space from the network.
        base_canvas = np.zeros((IMG, IMG), dtype=bool)
        base_canvas[0, :]  = True   # top wall
        base_canvas[-1, :] = True   # bottom wall
        base_canvas[:, 0]  = True   # left wall
        base_canvas[:, -1] = True   # right wall
        for verts_board in self._board.placed_polygons():
            verts_px = [(x * self._sx, y * self._sy) for x, y in verts_board]
            base_canvas |= _rasterize_polygon(verts_px, IMG)

        base_dist   = distance_transform_edt(~base_canvas).astype(np.float32)
        current_sdf = np.clip(base_dist, 0, self._sdf_clip_px) / self._sdf_clip_px

        # ── batch Rust preview for remaining parts × all rotations ─────────
        remaining_set = set(remaining_eps)
        flat_ids:  list[str]   = []
        flat_rots: list[float] = []
        for ep in range(N):
            if ep in remaining_set:
                gid = episode_geoms[ep]["id"]
                flat_ids.extend([gid] * R)
            else:
                flat_ids.extend(["__skip__"] * R)
            flat_rots.extend(rotations_rad)

        raw = self._board.preview_all_per_rotation(flat_ids, flat_rots)  # N×R entries

        # ── assemble result tensor ─────────────────────────────────────────
        result  = np.empty((N, 2 * R + 1, IMG, IMG), dtype=np.float32)
        scalars = np.zeros((N, R, 3), dtype=np.float32)   # [area_frac, bbox_w, bbox_h]
        result[:, 0] = current_sdf
        no_part_sdf = np.ones((IMG, IMG), dtype=np.float32)  # fallback isolated channel

        for ep in range(N):
            for r_idx in range(R):
                verts_board = raw[ep * R + r_idx]
                if verts_board is None or ep not in remaining_set:
                    result[ep, r_idx + 1]     = current_sdf
                    result[ep, R + 1 + r_idx] = no_part_sdf
                    # scalars stay 0 for non-fitting pairs
                else:
                    verts_px    = [(x * self._sx, y * self._sy) for x, y in verts_board]
                    part_canvas = _rasterize_polygon(verts_px, IMG)

                    # Board+part SDF: full EDT recompute with new part as obstacle.
                    # Previously this zeroed out part pixels in base_dist (fast but wrong:
                    # gradient around the new part was the old board's gradient, not a
                    # proper distance field including the new obstacle).
                    combined_dist = distance_transform_edt(
                        ~(base_canvas | part_canvas)
                    ).astype(np.float32)
                    result[ep, r_idx + 1] = (
                        np.clip(combined_dist, 0, self._sdf_clip_px) / self._sdf_clip_px
                    )

                    # Isolated part SDF: center the (rotated) part on a blank canvas
                    xs = [p[0] for p in verts_px]
                    ys = [p[1] for p in verts_px]
                    cx = (min(xs) + max(xs)) / 2.0
                    cy = (min(ys) + max(ys)) / 2.0
                    centered = [(x + half - cx, y + half - cy) for x, y in verts_px]
                    part_only = _rasterize_polygon(centered, IMG)
                    part_dist_arr = distance_transform_edt(~part_only).astype(np.float32)
                    result[ep, R + 1 + r_idx] = (
                        np.clip(part_dist_arr, 0, self._sdf_clip_px) / self._sdf_clip_px
                    )

                    # Scalar features: area, bbox width, bbox height (all in [0,1])
                    scalars[ep, r_idx, 0] = float(part_canvas.sum()) / (IMG * IMG)
                    scalars[ep, r_idx, 1] = (max(xs) - min(xs)) / IMG
                    scalars[ep, r_idx, 2] = (max(ys) - min(ys)) / IMG

        # Z-score each feature across N parts per rotation so small differences become
        # discriminatory.  When std≈0 (truly identical shapes) everything goes to 0,
        # which is correct — no information to convey.
        eps = 1e-6
        mean = scalars.mean(axis=0, keepdims=True)   # (1, R, 3)
        std  = scalars.std(axis=0, keepdims=True)    # (1, R, 3)
        scalars = (scalars - mean) / (std + eps)

        return result, scalars

    def preview_images_for_part(self, episode_id: int, rotations_rad: list[float]) -> np.ndarray:
        """
        Build (R+1, IMG_SIZE, IMG_SIZE) observation for a single part.

        Channel 0      : current board SDF.
        Channels 1..R+1: board SDF after placing part `episode_id` at each rotation.
        Falls back to current SDF if the part is already placed or doesn't fit.

        Used in two-phase eval: cheap N-call part selection followed by R-call
        rotation selection for the chosen part only (N+R instead of N×R).
        """
        from scipy.ndimage import distance_transform_edt

        R             = len(rotations_rad)
        IMG           = self.IMG_SIZE
        episode_geoms = self._episode_geoms()
        geom_id       = episode_geoms[episode_id]["id"]

        # Base board SDF (plate walls + placed parts)
        base_canvas = np.zeros((IMG, IMG), dtype=bool)
        base_canvas[0, :]  = True
        base_canvas[-1, :] = True
        base_canvas[:, 0]  = True
        base_canvas[:, -1] = True
        for verts_board in self._board.placed_polygons():
            verts_px = [(x * self._sx, y * self._sy) for x, y in verts_board]
            base_canvas |= _rasterize_polygon(verts_px, IMG)

        base_dist   = distance_transform_edt(~base_canvas).astype(np.float32)
        current_sdf = np.clip(base_dist, 0, self._sdf_clip_px) / self._sdf_clip_px

        raw = self._board.preview_all_per_rotation([geom_id] * R, rotations_rad)

        result    = np.empty((R + 1, IMG, IMG), dtype=np.float32)
        result[0] = current_sdf
        for r_idx, verts_board in enumerate(raw):
            if verts_board is None:
                result[r_idx + 1] = current_sdf
            else:
                verts_px    = [(x * self._sx, y * self._sy) for x, y in verts_board]
                part_canvas = _rasterize_polygon(verts_px, IMG)
                result_dist = base_dist.copy()
                result_dist[base_canvas | part_canvas] = 0.0
                result[r_idx + 1] = np.clip(result_dist, 0, self._sdf_clip_px) / self._sdf_clip_px

        return result

    def rollout_value(self) -> tuple[float, int]:
        """
        Run a greedy rollout on remaining parts, then restore the board.

        Returns (packing_density, n_placed):
          packing_density — placed_area / bbox_area after the greedy fill
          n_placed        — how many additional parts the greedy fill placed

        Used as a per-step value baseline during training.
        """
        return self._board.rollout_value()

    def packing_density(self) -> float:
        """Placed area / bounding-box area of all placed parts (0.0 if empty)."""
        return self._board.packing_density()

    def bbox_area(self) -> float:
        """Axis-aligned bounding box area of all placed parts."""
        return self._board.bbox_area()

    def n_placed(self) -> int:
        """Number of parts currently on the board."""
        return self._board.n_placed()

    def placed_polygons(self) -> list[list]:
        """Board-space polygon vertices for every placed part."""
        return self._board.placed_polygons()

    def place_remaining(self) -> int:
        """
        Greedily place all remaining parts (committing each).

        Returns the number of parts placed. Used during evaluation to get the
        greedy baseline density for a given episode configuration.
        """
        return self._board.place_remaining()

    def remaining_item_ids(self) -> list[int]:
        """
        Episode-space indices (0…N-1) of parts that have not yet been placed.

        Maps from string geometry IDs (Rust-side) back to integer episode IDs.
        """
        remaining_str = set(self._board.remaining_ids())
        episode_geoms = self._episode_geoms()
        return [
            ep_id
            for ep_id, geom in enumerate(episode_geoms)
            if geom["id"] in remaining_str
        ]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def preview_pair_images(self) -> np.ndarray:
        """
        Build (N, 2, IMG, IMG) pair images for the current step.

        Channel 0: current board SDF — same for all N candidates.
        Channel 1: board SDF after hypothetically placing part i.
                   Already-placed or unplaceable parts keep channel 1 = current SDF.

        Uses the min-distance trick: EDT runs once on the current board and once
        per candidate part (single polygon), then combined as
            result_dist = min(base_dist, part_dist)
        avoiding a full multi-polygon EDT re-run for each of the N candidates.

        Returns:
            (N, 2, IMG_SIZE, IMG_SIZE) float32
        """
        from scipy.ndimage import distance_transform_edt

        N             = len(self.current_lib_ids)
        remaining_eps = self.remaining_item_ids()
        episode_geoms = self._episode_geoms()

        # String IDs of remaining parts (for the Rust batch preview call)
        remaining_str = [episode_geoms[ep_id]["id"] for ep_id in remaining_eps]
        preview_verts = self._board.preview_all(remaining_str)  # list[list|None]
        # Map string_id → board-space vertices (or None)
        preview_map: dict[str, list | None] = dict(zip(remaining_str, preview_verts))

        # --- base canvas: plate walls + all currently placed parts ---
        S = self.IMG_SIZE
        base_canvas = np.zeros((S, S), dtype=bool)
        base_canvas[0, :]  = True
        base_canvas[-1, :] = True
        base_canvas[:, 0]  = True
        base_canvas[:, -1] = True
        for verts_board in self._board.placed_polygons():
            verts_px = [(x * self._sx, y * self._sy) for x, y in verts_board]
            base_canvas |= _rasterize_polygon(verts_px, S)

        # --- current board SDF (channel 0) ---
        base_dist   = distance_transform_edt(~base_canvas).astype(np.float32)
        current_sdf = np.clip(base_dist, 0, self._sdf_clip_px) / self._sdf_clip_px

        # --- result tensor: default both channels to current SDF ---
        result = np.empty((N, 2, self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        result[:, 0] = current_sdf
        result[:, 1] = current_sdf  # fallback for placed / unplaceable

        for ep_id in range(N):
            geom_id     = episode_geoms[ep_id]["id"]
            verts_board = preview_map.get(geom_id)
            if verts_board is None:
                continue
            verts_px    = [(x * self._sx, y * self._sy) for x, y in verts_board]
            part_canvas = _rasterize_polygon(verts_px, self.IMG_SIZE)
            # Same fast approximation as preview_images_per_rotation: copy base_dist
            # and zero-fill the placement footprint. Avoids a per-part EDT call
            # (the per-part EDT was the eval bottleneck and also inconsistent with
            # how training observations are computed).
            result_dist = base_dist.copy()
            result_dist[base_canvas | part_canvas] = 0.0
            result[ep_id, 1] = np.clip(result_dist, 0, self._sdf_clip_px) / self._sdf_clip_px

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _episode_geoms(self) -> list[dict]:
        return [self._library[i] for i in self.current_lib_ids]

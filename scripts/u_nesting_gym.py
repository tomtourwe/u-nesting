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
            self._library: list[dict] = raw
        elif isinstance(raw, dict) and "items" in raw:
            # Convert sparrow format — orientations come from CLI, not the library
            self._library = [
                {"id": item.get("label", str(i)), "polygon": item["shape"]["data"]}
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

        # --- base canvas: all currently placed parts ---
        base_canvas = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=bool)
        for verts_board in self._board.placed_polygons():
            verts_px = [(x * self._sx, y * self._sy) for x, y in verts_board]
            base_canvas |= _rasterize_polygon(verts_px, self.IMG_SIZE)

        # --- current board SDF (channel 0) ---
        if base_canvas.any():
            base_dist   = distance_transform_edt(~base_canvas).astype(np.float32)
            current_sdf = np.clip(base_dist, 0, self._sdf_clip_px) / self._sdf_clip_px
        else:
            base_dist   = np.full((self.IMG_SIZE, self.IMG_SIZE), np.inf, dtype=np.float32)
            current_sdf = np.ones((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)

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
            part_dist   = distance_transform_edt(~part_canvas).astype(np.float32)
            result_dist = np.minimum(base_dist, part_dist)
            result_dist[base_canvas | part_canvas] = 0.0
            result[ep_id, 1] = np.clip(result_dist, 0, self._sdf_clip_px) / self._sdf_clip_px

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _episode_geoms(self) -> list[dict]:
        return [self._library[i] for i in self.current_lib_ids]

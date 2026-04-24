//! Stateful 2D nesting board for incremental placement (e.g. RL agents).

use std::collections::HashMap;

use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::Config;

use crate::boundary::Boundary2D;
use crate::geometry::Geometry2D;
use crate::nester::Nester2D;
use crate::nfp::PlacedGeometry;
use u_nesting_core::Result;

/// A lightweight snapshot entry: geometry id + placement coordinates.
/// Does not clone polygon data — geometry is reconstructed from the board's map on restore.
#[derive(Debug, Clone)]
pub struct SnapEntry {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub rotation: f64,
}

/// Placement result returned by `Board2D::place`.
#[derive(Debug, Clone)]
pub struct PlacementInfo {
    pub geometry_id: String,
    pub x: f64,
    pub y: f64,
    pub rotation: f64,
}

/// Stateful 2D nesting board.
///
/// Holds a fixed boundary and a growing list of placed geometries. The agent
/// calls `place()` to add one part at a time; the engine finds the best valid
/// position. State can be saved and restored cheaply for rollouts.
pub struct Board2D {
    nester: Nester2D,
    boundary: Boundary2D,
    boundary_measure: f64,
    placed: Vec<PlacedGeometry>,
    geom_map: HashMap<String, Geometry2D>,
    episode_ids: Vec<String>,
    sample_step: f64,
}

impl Board2D {
    /// Creates a new board.
    ///
    /// `geometries` is the full set of parts that will ever be placed on this
    /// board. It is used to pre-compute a good grid sample step and to look up
    /// geometry definitions during `restore`.
    pub fn new(boundary: Boundary2D, geometries: &[Geometry2D], config: Config) -> Self {
        let nester = Nester2D::new(config);
        let sample_step = nester.compute_sample_step(geometries);
        let boundary_measure = boundary.measure();
        let geom_map: HashMap<String, Geometry2D> = geometries
            .iter()
            .map(|g| (g.id().to_string(), g.clone()))
            .collect();
        let episode_ids: Vec<String> = geometries.iter().map(|g| g.id().to_string()).collect();

        Self {
            nester,
            boundary,
            boundary_measure,
            placed: Vec::new(),
            geom_map,
            episode_ids,
            sample_step,
        }
    }

    /// Attempts to place the geometry with the given id onto the board.
    ///
    /// The engine finds the best valid position across all allowed rotations.
    /// Returns `Some(PlacementInfo)` on success, `None` if the part doesn't fit.
    pub fn place(&mut self, id: &str) -> Result<Option<PlacementInfo>> {
        let geom = match self.geom_map.get(id) {
            Some(g) => g.clone(),
            None => return Ok(None),
        };

        match self
            .nester
            .place_one_part(&geom, &self.placed, &self.boundary, self.sample_step)?
        {
            Some((x, y, rotation)) => {
                self.placed
                    .push(PlacedGeometry::new(geom.clone(), (x, y), rotation));
                Ok(Some(PlacementInfo {
                    geometry_id: id.to_string(),
                    x,
                    y,
                    rotation,
                }))
            }
            None => Ok(None),
        }
    }

    /// Removes the last placed geometry. Returns `true` if something was removed.
    pub fn undo(&mut self) -> bool {
        self.placed.pop().is_some()
    }

    /// Removes all placed geometries.
    pub fn reset(&mut self) {
        self.placed.clear();
    }

    /// Returns a lightweight snapshot of the current board state.
    ///
    /// Only stores ids and positions — no polygon data is cloned.
    pub fn snapshot(&self) -> Vec<SnapEntry> {
        self.placed
            .iter()
            .map(|p| SnapEntry {
                id: p.geometry.id().to_string(),
                x: p.position.0,
                y: p.position.1,
                rotation: p.rotation,
            })
            .collect()
    }

    /// Restores the board to a previously snapshotted state.
    ///
    /// Entries not found in the geometry map are silently skipped.
    pub fn restore(&mut self, snap: &[SnapEntry]) {
        self.placed = snap
            .iter()
            .filter_map(|e| {
                self.geom_map.get(&e.id).map(|g| {
                    PlacedGeometry::new(g.clone(), (e.x, e.y), e.rotation)
                })
            })
            .collect();
    }

    /// Returns the current fill ratio (0.0–1.0).
    pub fn utilization(&self) -> f64 {
        if self.boundary_measure == 0.0 {
            return 0.0;
        }
        let placed_area: f64 = self.placed.iter().map(|p| p.geometry.measure()).sum();
        placed_area / self.boundary_measure
    }

    /// Returns a reference to the current placed geometries.
    pub fn placements(&self) -> &[PlacedGeometry] {
        &self.placed
    }

    /// Returns the board-space polygon vertices for every placed part.
    ///
    /// Each entry is the exterior ring of one placed geometry, already
    /// rotated and translated to its position on the board.
    pub fn placed_polygons(&self) -> Vec<Vec<(f64, f64)>> {
        self.placed
            .iter()
            .map(|p| p.translated_exterior())
            .collect()
    }

    /// Number of parts currently placed.
    pub fn placed_count(&self) -> usize {
        self.placed.len()
    }

    /// Alias for `placed_count` — matches the sparrow/gym naming convention.
    pub fn n_placed(&self) -> usize {
        self.placed.len()
    }

    /// Returns the ordered list of geometry IDs that have not yet been placed.
    ///
    /// Order matches the episode insertion order (i.e. the order `geometries`
    /// was passed to `new()`), which ensures deterministic LBF rollouts.
    pub fn remaining_ids(&self) -> Vec<String> {
        let placed_ids: std::collections::HashSet<String> =
            self.placed.iter().map(|p| p.geometry.id().to_string()).collect();
        self.episode_ids
            .iter()
            .filter(|id| !placed_ids.contains(*id))
            .cloned()
            .collect()
    }

    /// For each geometry id in `ids`, speculatively place it via LBF (no commit)
    /// and return the board-space polygon vertices, or `None` if it doesn't fit.
    ///
    /// The current `placed` state is not modified.
    pub fn lbf_preview_all(&self, ids: &[String]) -> Vec<Option<Vec<(f64, f64)>>> {
        ids.iter()
            .map(|id| {
                let geom = self.geom_map.get(id)?;
                match self
                    .nester
                    .place_one_part(geom, &self.placed, &self.boundary, self.sample_step)
                {
                    Ok(Some((x, y, rotation))) => {
                        let pg = PlacedGeometry::new(geom.clone(), (x, y), rotation);
                        Some(pg.translated_exterior())
                    }
                    _ => None,
                }
            })
            .collect()
    }

    /// Run a full LBF greedy rollout on the remaining parts, then restore the
    /// board to its state before the call.
    ///
    /// Returns `(packing_density, n_placed)`:
    /// - `packing_density` — `placed_area / bbox_area` after filling in the remaining parts
    /// - `n_placed` — how many additional parts the greedy rollout managed to place
    ///
    /// Used as a per-step value baseline during RL training.
    pub fn lbf_rollout_value(&mut self) -> (f64, usize) {
        let snap = self.snapshot();
        let remaining = self.remaining_ids();
        let mut n_placed = 0usize;
        for id in &remaining {
            if let Ok(Some(_)) = self.place(id) {
                n_placed += 1;
            }
        }
        let density = self.packing_density();
        self.restore(&snap);
        (density, n_placed)
    }

    /// Place all remaining parts via LBF (committing each successful placement).
    ///
    /// Returns the number of parts that were successfully placed.
    pub fn lbf_place_all(&mut self) -> usize {
        let remaining = self.remaining_ids();
        let mut n_placed = 0usize;
        for id in &remaining {
            if let Ok(Some(_)) = self.place(id) {
                n_placed += 1;
            }
        }
        n_placed
    }

    /// `placed_area / bbox_area` of all placed parts (0.0 if nothing is placed).
    ///
    /// This is the sparrow-style packing density metric: how efficiently the
    /// parts fill their own bounding box, rather than the full board area.
    pub fn packing_density(&self) -> f64 {
        let bbox = self.bbox_area();
        if bbox == 0.0 {
            return 0.0;
        }
        let placed_area: f64 = self.placed.iter().map(|p| p.geometry.measure()).sum();
        placed_area / bbox
    }

    /// Axis-aligned bounding box area of all placed parts (0.0 if nothing is placed).
    pub fn bbox_area(&self) -> f64 {
        if self.placed.is_empty() {
            return 0.0;
        }
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for verts in self.placed_polygons() {
            for (x, y) in verts {
                if x < min_x { min_x = x; }
                if y < min_y { min_y = y; }
                if x > max_x { max_x = x; }
                if y > max_y { max_y = y; }
            }
        }
        if max_x <= min_x || max_y <= min_y {
            return 0.0;
        }
        (max_x - min_x) * (max_y - min_y)
    }
}

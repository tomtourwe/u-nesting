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
    /// Fine grid step used for committed placements (accurate board state).
    sample_step: f64,
    /// Coarse grid step used for rollout/preview operations (faster, value-only).
    rollout_sample_step: f64,
}

impl Board2D {
    /// Creates a new board with default rollout step multiplier (3×).
    pub fn new(boundary: Boundary2D, geometries: &[Geometry2D], config: Config) -> Self {
        Self::new_with_rollout_multiplier(boundary, geometries, config, 3.0)
    }

    /// Creates a new board with an explicit rollout step multiplier.
    ///
    /// `rollout_step_multiplier` scales the grid step used inside
    /// `rollout_value`, `place_remaining`, and `preview_all`.
    /// A larger multiplier means fewer grid points → faster rollouts at the
    /// cost of slightly coarser speculative placements (value estimates only).
    /// Committed `place()` calls always use the fine grid.
    pub fn new_with_rollout_multiplier(
        boundary: Boundary2D,
        geometries: &[Geometry2D],
        config: Config,
        rollout_step_multiplier: f64,
    ) -> Self {
        let nester = Nester2D::new(config);
        let sample_step = nester.compute_sample_step(geometries);
        let rollout_sample_step = (sample_step * rollout_step_multiplier).clamp(sample_step, 50.0);
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
            rollout_sample_step,
        }
    }

    /// Attempts to place the geometry with the given id onto the board.
    ///
    /// Uses the fine grid step — positions are accurate for board-state quality.
    /// Returns `Some(PlacementInfo)` on success, `None` if the part doesn't fit.
    pub fn place(&mut self, id: &str) -> Result<Option<PlacementInfo>> {
        let geom = match self.geom_map.get(id) {
            Some(g) => g.clone(),
            None => return Ok(None),
        };
        match self.nester.place_one_part(&geom, &self.placed, &self.boundary, self.sample_step)? {
            Some((x, y, rotation)) => {
                self.placed.push(PlacedGeometry::new(geom.clone(), (x, y), rotation));
                Ok(Some(PlacementInfo { geometry_id: id.to_string(), x, y, rotation }))
            }
            None => Ok(None),
        }
    }

    /// Like `place()` but uses the coarser rollout grid step.
    ///
    /// NFP vertex candidates are still included (for placement quality) but the
    /// supplementary grid is sparser. Only used for value-estimation rollouts;
    /// committed placements always use the fine-grid `place()`.
    fn place_fast(&mut self, id: &str) -> Result<Option<PlacementInfo>> {
        let geom = match self.geom_map.get(id) {
            Some(g) => g.clone(),
            None => return Ok(None),
        };

        match self.nester.place_one_part(
            &geom,
            &self.placed,
            &self.boundary,
            self.rollout_sample_step,
        )? {
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

    /// Start a new episode with the given geometry IDs (a subset of the board's library).
    ///
    /// Clears all placed parts and sets the active episode to the given IDs.
    /// The underlying `Nester2D` — and crucially its NFP cache — is preserved,
    /// so cached NFPs from previous episodes are reused in future ones.
    ///
    /// IDs not present in the geometry map are silently ignored.
    pub fn start_episode(&mut self, ids: &[String]) {
        self.placed.clear();
        self.episode_ids = ids
            .iter()
            .filter(|id| self.geom_map.contains_key(*id))
            .cloned()
            .collect();
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
    pub fn remaining_ids(&self) -> Vec<String> {
        let placed_ids: std::collections::HashSet<String> =
            self.placed.iter().map(|p| p.geometry.id().to_string()).collect();
        self.episode_ids
            .iter()
            .filter(|id| !placed_ids.contains(*id))
            .cloned()
            .collect()
    }

    /// Place part `id` at a fixed `rotation` (radians) using the fine grid.
    ///
    /// The NFP engine finds the best (x, y) for that rotation only.
    /// Returns `Some(PlacementInfo)` on success, `None` if the part doesn't fit.
    pub fn place_with_rotation(
        &mut self,
        id: &str,
        rotation: f64,
    ) -> Result<Option<PlacementInfo>> {
        let geom = match self.geom_map.get(id) {
            Some(g) => g.clone(),
            None => return Ok(None),
        };
        match self.nester.place_one_part_at_rotation(
            &geom,
            &self.placed,
            &self.boundary,
            self.sample_step,
            rotation,
        )? {
            Some((x, y, r)) => {
                self.placed.push(PlacedGeometry::new(geom.clone(), (x, y), r));
                Ok(Some(PlacementInfo { geometry_id: id.to_string(), x, y, rotation: r }))
            }
            None => Ok(None),
        }
    }

    /// Speculatively place each `(id, rotation)` pair (without committing) and
    /// return board-space polygon vertices, or `None` if the part doesn't fit
    /// at that rotation.
    ///
    /// `ids` and `rotations` are parallel slices of the same length (N×R).
    /// Uses the coarse rollout grid. Evaluated in parallel when `parallel` is enabled.
    pub fn preview_all_per_rotation(
        &self,
        ids: &[String],
        rotations: &[f64],
    ) -> Vec<Option<Vec<(f64, f64)>>> {
        let step = self.rollout_sample_step;
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            ids.par_iter()
                .zip(rotations.par_iter())
                .map(|(id, &rotation)| {
                    let geom = self.geom_map.get(id)?;
                    match self.nester.place_one_part_at_rotation(
                        geom,
                        &self.placed,
                        &self.boundary,
                        step,
                        rotation,
                    ) {
                        Ok(Some((x, y, r))) => {
                            let pg = PlacedGeometry::new(geom.clone(), (x, y), r);
                            Some(pg.translated_exterior())
                        }
                        _ => None,
                    }
                })
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            ids.iter()
                .zip(rotations.iter())
                .map(|(id, &rotation)| {
                    let geom = self.geom_map.get(id)?;
                    match self.nester.place_one_part_at_rotation(
                        geom,
                        &self.placed,
                        &self.boundary,
                        step,
                        rotation,
                    ) {
                        Ok(Some((x, y, r))) => {
                            let pg = PlacedGeometry::new(geom.clone(), (x, y), r);
                            Some(pg.translated_exterior())
                        }
                        _ => None,
                    }
                })
                .collect()
        }
    }

    /// For each geometry id in `ids`, speculatively place it (no commit) and
    /// return the board-space polygon vertices, or `None` if it doesn't fit.
    ///
    /// Uses the coarse rollout grid — fast enough for per-step observation
    /// building. Each candidate is evaluated in parallel when the `parallel`
    /// feature is enabled.
    pub fn preview_all(&self, ids: &[String]) -> Vec<Option<Vec<(f64, f64)>>> {
        let step = self.rollout_sample_step;
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            ids.par_iter()
                .map(|id| {
                    let geom = self.geom_map.get(id)?;
                    match self.nester.place_one_part(geom, &self.placed, &self.boundary, step) {
                        Ok(Some((x, y, rotation))) => {
                            let pg = PlacedGeometry::new(geom.clone(), (x, y), rotation);
                            Some(pg.translated_exterior())
                        }
                        _ => None,
                    }
                })
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            ids.iter()
                .map(|id| {
                    let geom = self.geom_map.get(id)?;
                    match self.nester.place_one_part(geom, &self.placed, &self.boundary, step) {
                        Ok(Some((x, y, rotation))) => {
                            let pg = PlacedGeometry::new(geom.clone(), (x, y), rotation);
                            Some(pg.translated_exterior())
                        }
                        _ => None,
                    }
                })
                .collect()
        }
    }

    /// Snapshot → coarse greedy rollout on remaining parts → restore.
    ///
    /// Uses the coarse grid for speed. Returns `(packing_density, n_placed)`.
    /// Used as a per-step value baseline during RL training.
    pub fn rollout_value(&mut self) -> (f64, usize) {
        let snap = self.snapshot();
        let mut remaining = self.remaining_ids();
        remaining.sort_by(|a, b| {
            let area_a = self.geom_map.get(a).map_or(0.0, |g| g.measure());
            let area_b = self.geom_map.get(b).map_or(0.0, |g| g.measure());
            area_b.partial_cmp(&area_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut n_placed = 0usize;
        for id in &remaining {
            if let Ok(Some(_)) = self.place_fast(id) {
                n_placed += 1;
            }
        }
        let density = self.packing_density();
        self.restore(&snap);
        (density, n_placed)
    }

    /// Place all remaining parts via the coarse grid (committing each).
    ///
    /// Used for evaluation baselines. Returns the number placed.
    pub fn place_remaining(&mut self) -> usize {
        let mut remaining = self.remaining_ids();
        remaining.sort_by(|a, b| {
            let area_a = self.geom_map.get(a).map_or(0.0, |g| g.measure());
            let area_b = self.geom_map.get(b).map_or(0.0, |g| g.measure());
            area_b.partial_cmp(&area_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut n_placed = 0usize;
        for id in &remaining {
            if let Ok(Some(_)) = self.place_fast(id) {
                n_placed += 1;
            }
        }
        n_placed
    }

    /// Returns NFP cache statistics: `(hits, misses, size)`.
    pub fn cache_stats(&self) -> (usize, usize, usize) {
        self.nester.cache_stats()
    }

    /// `placed_area / bbox_area` of all placed parts (0.0 if nothing is placed).
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

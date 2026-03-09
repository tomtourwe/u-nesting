//! WASM-compatible timing abstraction.
//!
//! On native targets, uses [`std::time::Instant`] for wall-clock timing.
//! On WASM (`wasm32`), provides a no-op timer that always reports zero elapsed time.
//! This allows algorithms to compile and run on WASM without panicking,
//! while still supporting time-based termination on native platforms.

#[cfg(not(target_arch = "wasm32"))]
mod inner {
    use std::time::{Duration, Instant};

    /// A wall-clock timer backed by [`Instant`] on native targets.
    #[derive(Debug, Clone, Copy)]
    pub struct Timer(Instant);

    impl Timer {
        /// Starts the timer.
        pub fn now() -> Self {
            Timer(Instant::now())
        }

        /// Returns the elapsed time since the timer was started.
        pub fn elapsed(&self) -> Duration {
            self.0.elapsed()
        }

        /// Returns elapsed time in milliseconds.
        pub fn elapsed_ms(&self) -> u64 {
            self.0.elapsed().as_millis() as u64
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod inner {
    use std::time::Duration;

    /// A no-op timer for WASM targets.
    ///
    /// Always reports zero elapsed time. Algorithms terminate via
    /// iteration limits instead of wall-clock time on WASM.
    #[derive(Debug, Clone, Copy)]
    pub struct Timer;

    impl Timer {
        /// No-op: returns immediately.
        pub fn now() -> Self {
            Timer
        }

        /// Always returns [`Duration::ZERO`].
        pub fn elapsed(&self) -> Duration {
            Duration::ZERO
        }

        /// Always returns `0`.
        pub fn elapsed_ms(&self) -> u64 {
            0
        }
    }
}

pub use inner::Timer;

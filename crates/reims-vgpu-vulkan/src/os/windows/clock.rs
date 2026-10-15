//! Windows clocks: interrupt time includes host sleep, and wall time is the Unix epoch.

use super::api;
use reims_vgpu_core::platform::os::{Clock, Deadline, MonoTime, TimedOut, WallTime};
use reims_vgpu_core::sync::{Condvar, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The host's clocks and timed waits.
#[derive(Debug, Default)]
pub struct WindowsClock;

impl WindowsClock {
    /// The process's clocks.
    pub fn new() -> Self {
        Self
    }
    fn remaining(&self, deadline: Deadline) -> Duration {
        match deadline {
            Deadline::Mono(t) => t.saturating_duration_since(self.now_mono()),
            Deadline::Wall(t) => t.saturating_duration_since(self.now_wall()),
        }
    }
}

impl Clock for WindowsClock {
    fn now_mono(&self) -> MonoTime {
        let mut ticks = 0;
        // SAFETY: a live output parameter; interrupt time is in 100 ns units.
        unsafe { api::QueryInterruptTimePrecise(&mut ticks) };
        MonoTime::from_nanos(ticks.saturating_mul(100))
    }
    #[expect(
        clippy::disallowed_methods,
        reason = "the Windows Clock implementation reads std's Windows wall clock"
    )]
    fn now_wall(&self) -> WallTime {
        WallTime::from_nanos(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)),
        )
    }
    fn wait_until<T>(
        &self,
        cv: &Condvar,
        g: &mut MutexGuard<'_, T>,
        deadline: Deadline,
    ) -> TimedOut {
        let duration = self.remaining(deadline);
        if !duration.is_zero() {
            let _ = cv.wait_for(g, duration);
        }
        TimedOut(deadline.reached(self))
    }
    fn wait<T>(&self, cv: &Condvar, g: &mut MutexGuard<'_, T>) {
        cv.wait(g);
    }
    fn sleep_until(&self, deadline: Deadline) {
        loop {
            let duration = self.remaining(deadline);
            if duration.is_zero() {
                return;
            }
            std::thread::sleep(duration);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_clocks_advance_and_past_deadlines_do_not_wait() {
        let clock = WindowsClock::new();
        let before = clock.now_mono();
        std::thread::sleep(Duration::from_millis(2));
        assert!(clock.now_mono().saturating_duration_since(before) >= Duration::from_millis(2));
        assert!(clock.now_wall().as_nanos() > 1_000_000_000_000_000_000);
        let lock = reims_vgpu_core::sync::Mutex::new(());
        let cv = Condvar::new();
        assert!(
            clock
                .wait_until(&cv, &mut lock.lock(), Deadline::Mono(before))
                .0
        );
    }
}

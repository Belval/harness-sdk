//! Watchdogs: a swarm that deadlocks, or hangs before it is up, hangs silently, which a supervisor
//! cannot tell from a busy one. They turn the hang into an exit, so `swarmctl` restarts the run (or
//! rolls back a binary that hangs right after a reload).

use std::fmt::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Exit code for a detected deadlock.
pub const DEADLOCK_EXIT_CODE: i32 = 70;

/// Exit code for a swarm that was not up within [`STARTUP_TIMEOUT`].
pub const STARTUP_HANG_EXIT_CODE: i32 = 71;

/// How long a swarm may take from start to its first TUI frame (or, headless, to running).
pub const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);

static READY: AtomicBool = AtomicBool::new(false);

/// Mark the swarm as up (its first frame is drawn), which ends [`watch_startup`]'s watch.
pub fn ready() {
    READY.store(true, Ordering::SeqCst);
}

/// Whether [`ready`] was called.
pub fn is_ready() -> bool {
    READY.load(Ordering::SeqCst)
}

/// Call `on_hang` on a background thread unless `ready` turns true within `timeout`.
pub fn watch_startup(
    timeout: Duration,
    ready: impl Fn() -> bool + Send + 'static,
    on_hang: impl FnOnce() + Send + 'static,
) {
    let spawned = std::thread::Builder::new()
        .name("startup-watchdog".into())
        .spawn(move || {
            let deadline = Instant::now() + timeout;
            while !ready() {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    on_hang();
                    return;
                }
                std::thread::sleep(left.min(Duration::from_millis(100)));
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "startup watchdog not started");
    }
}

/// Check for deadlocked `parking_lot` locks every `interval` on a background thread, and call
/// `on_deadlock` with a report (threads and backtraces) the first time one is found.
pub fn spawn(interval: Duration, on_deadlock: impl FnOnce(String) + Send + 'static) {
    let spawned = std::thread::Builder::new()
        .name("deadlock-watchdog".into())
        .spawn(move || loop {
            std::thread::sleep(interval);
            if let Some(report) = check() {
                on_deadlock(report);
                return;
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "deadlock watchdog not started");
    }
}

/// A report of the deadlocked threads, if any.
pub fn check() -> Option<String> {
    let cycles = parking_lot::deadlock::check_deadlock();
    if cycles.is_empty() {
        return None;
    }
    let mut report = format!("{} deadlock cycle(s) detected\n", cycles.len());
    for (index, threads) in cycles.iter().enumerate() {
        for thread in threads {
            let _ = writeln!(
                report,
                "cycle {index}, thread {:?}:\n{:?}",
                thread.thread_id(),
                thread.backtrace()
            );
        }
    }
    Some(report)
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::sync::Arc;

    use super::*;

    #[test]
    fn the_startup_watchdog_fires_only_when_the_swarm_is_not_up_in_time() {
        let (hung, hangs) = mpsc::channel();
        watch_startup(
            Duration::from_millis(50),
            || false,
            move || hung.send(()).unwrap(),
        );
        assert_eq!(hangs.recv_timeout(Duration::from_secs(5)), Ok(()));

        let up = Arc::new(AtomicBool::new(false));
        let (hung, hangs) = mpsc::channel();
        let flag = up.clone();
        watch_startup(
            Duration::from_millis(300),
            move || flag.load(Ordering::SeqCst),
            move || hung.send(()).unwrap(),
        );
        up.store(true, Ordering::SeqCst);
        assert!(hangs.recv_timeout(Duration::from_secs(1)).is_err());
    }
}

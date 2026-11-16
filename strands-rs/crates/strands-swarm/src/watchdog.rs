//! Deadlock watchdog: a swarm that deadlocks hangs silently, which a supervisor cannot tell from
//! a busy one. The watchdog turns a deadlock into an exit, so `swarmctl` restarts the run (or rolls
//! back a binary that deadlocks right after a reload).

use std::fmt::Write;
use std::time::Duration;

/// Exit code for a detected deadlock.
pub const DEADLOCK_EXIT_CODE: i32 = 70;

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

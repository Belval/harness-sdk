//! The deadlock watchdog, in its own test binary: the deadlocked threads it provokes never end.

use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

use parking_lot::Mutex;
use strands_swarm::watchdog;

#[test]
fn reports_a_lock_order_deadlock() {
    assert!(watchdog::check().is_none());
    let (sent, reported) = mpsc::channel();
    watchdog::spawn(Duration::from_millis(50), move |report| {
        let _ = sent.send(report);
    });

    let a = Arc::new(Mutex::new(()));
    let b = Arc::new(Mutex::new(()));
    let barrier = Arc::new(Barrier::new(2));
    for (first, second) in [(a.clone(), b.clone()), (b, a)] {
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            let _first = first.lock();
            barrier.wait();
            let _second = second.lock();
        });
    }

    let report = reported.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(report.starts_with("1 deadlock cycle(s) detected"), "{report}");
}

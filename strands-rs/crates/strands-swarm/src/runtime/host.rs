//! Host protection: while the host is overloaded (1-minute load average above
//! [`Swarm::max_load`], or available memory below [`SwarmConfig::min_free_mem`]), new model
//! calls, background jobs and agents spawned by agents wait; while its disk is low, agents'
//! worktree spawns and land jobs too. Nothing already running is held.

use super::*;

/// How long a reading of the host stays fresh, and how often a held caller looks again.
pub const HOST_POLL: Duration = Duration::from_secs(1);

/// One reading of the host.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostStats {
    /// 1-minute load average.
    pub load: f64,
    /// Memory available for new work (`MemAvailable`), in bytes.
    pub mem_available: u64,
}

/// Reads the host; `None` when unknown, which holds nothing.
pub type HostProbe = Arc<dyn Fn() -> Option<HostStats> + Send + Sync>;

/// The host's load and available memory, from `/proc/loadavg` and `/proc/meminfo`.
pub fn host_stats() -> Option<HostStats> {
    let loadavg = std::fs::read_to_string("/proc/loadavg").ok()?;
    let load = loadavg.split_whitespace().next()?.parse().ok()?;
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mem_available = meminfo_bytes(&meminfo, "MemAvailable")?;
    Some(HostStats { load, mem_available })
}

/// The default `--max-load`: twice the CPUs.
pub fn default_max_load() -> f64 {
    let cpus = std::thread::available_parallelism().map_or(1, usize::from);
    2.0 * cpus as f64
}

/// The default `--min-free-mem`: 5% of the host's memory, at most 8 GB (`0`, off, when unknown).
pub fn default_min_free_mem() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| meminfo_bytes(&text, "MemTotal"))
        .map_or(0, |total| (total / 20).min(8 << 30))
}

/// `key`'s value in `/proc/meminfo` text (`MemAvailable:   123 kB`), in bytes.
fn meminfo_bytes(text: &str, key: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.split(':').next() == Some(key))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kb.checked_mul(1024)
}

/// The cached reading, the runtime limits and who is held.
#[derive(Default)]
pub(super) struct HostGate {
    reading: Mutex<Option<(tokio::time::Instant, Option<HostStats>)>>,
    /// Fired on every fresh reading and limit change: held callers look at the cache again.
    pub(super) wake: tokio::sync::Notify,
    /// Set with [`Swarm::set_max_load`]; overrides the config.
    pub(super) max_load: Mutex<Option<f64>>,
    /// Set with [`Swarm::set_min_free_mem`]; overrides the config.
    pub(super) min_free_mem: Mutex<Option<u64>>,
    /// Agents held now, with why.
    pub(super) held: Mutex<BTreeMap<String, String>>,
    /// Why worktree spawns and land jobs wait: space is low ([`Swarm::guard_disk`]).
    pub(super) disk_low: Mutex<Option<String>>,
    /// The least free bytes and inodes over the disks the guard watches, as it last read them.
    pub(super) disk_free: Mutex<Option<(u64, Option<u64>)>>,
    /// The fitted decline of the free bytes of the filesystem `disk_free` reads, once known.
    pub(super) disk_trend: Mutex<Option<super::trend::Estimate>>,
}

/// Clears an agent's hold when its wait ends, however it ends.
struct Held<'a>(&'a Swarm, &'a str);

impl Drop for Held<'_> {
    fn drop(&mut self) {
        if self.0.inner.host.held.lock().remove(self.1).is_some() {
            self.0.emit(SwarmEvent::HostHeld {
                id: self.1.to_owned(),
                reason: None,
            });
        }
    }
}

impl Swarm {
    /// The load limit in force (`0`: none): the last [`Swarm::set_max_load`], else the config's.
    pub fn max_load(&self) -> f64 {
        self.inner
            .host
            .max_load
            .lock()
            .unwrap_or(self.inner.config.max_load)
    }

    /// Change the load limit for this run (kept across restarts, ahead of `--max-load`);
    /// `Some(0.0)` lifts it and `None` returns to the configured one. Held callers look again
    /// at once.
    pub fn set_max_load(&self, limit: Option<f64>) {
        *self.inner.host.max_load.lock() = limit;
        self.inner.host.wake.notify_waiters();
        self.persist();
        self.journal_event("max_load", serde_json::json!({ "limit": self.max_load() }));
    }

    /// The free-memory floor in force, in bytes (`0`: none): the last
    /// [`Swarm::set_min_free_mem`], else the config's.
    pub fn min_free_mem(&self) -> u64 {
        self.inner
            .host
            .min_free_mem
            .lock()
            .unwrap_or(self.inner.config.min_free_mem)
    }

    /// Change the free-memory floor for this run (kept across restarts, ahead of
    /// `--min-free-mem`); `Some(0)` lifts it and `None` returns to the configured one.
    pub fn set_min_free_mem(&self, floor: Option<u64>) {
        *self.inner.host.min_free_mem.lock() = floor;
        self.inner.host.wake.notify_waiters();
        self.persist();
        self.journal_event(
            "min_free_mem",
            serde_json::json!({ "bytes": self.min_free_mem() }),
        );
    }

    /// The host's load and memory, read at most once per [`HOST_POLL`]. A fresh reading wakes the
    /// held callers, so one caller's poll serves them all.
    pub fn host(&self) -> Option<HostStats> {
        let mut reading = self.inner.host.reading.lock();
        match *reading {
            Some((at, stats)) if at.elapsed() < HOST_POLL => stats,
            _ => {
                let stats = (self.inner.config.host_probe)();
                *reading = Some((tokio::time::Instant::now(), stats));
                self.inner.host.wake.notify_waiters();
                stats
            }
        }
    }

    /// Why new work must wait now, e.g. `host load 180/128`; `None` when it need not.
    pub fn host_busy(&self) -> Option<String> {
        let stats = self.host()?;
        let max_load = self.max_load();
        if max_load > 0.0 && stats.load > max_load {
            return Some(format!(
                "host load {:.0}/{max_load:.0} (/max-load to change)",
                stats.load
            ));
        }
        let floor = self.min_free_mem();
        (stats.mem_available < floor).then(|| {
            format!(
                "host memory {} available, below {} (/min-free-mem to change)",
                crate::gc::human(stats.mem_available),
                crate::gc::human(floor)
            )
        })
    }

    /// Wait while [`Self::host_busy`] (with `disk`, also while disk space is low:
    /// [`Swarm::guard_disk`]), published as [`SwarmEvent::HostHeld`] for agent `id`. Errors with
    /// [`Error::Cancelled`] when `cancel` fires or the swarm shuts down first. An agent in the
    /// priority lane ([`Swarm::is_priority`]) waits for disk space only.
    pub(super) async fn wait_for_host(&self, id: &str, disk: bool, cancel: &CancellationToken) -> Result<()> {
        let busy = || {
            let host = || (!self.is_priority(id)).then(|| self.host_busy()).flatten();
            host().or_else(|| disk.then(|| self.disk_low()).flatten())
        };
        let _held = Held(self, id);
        let wake = &self.inner.host.wake;
        loop {
            // Registered before the check, so a reading in between is not missed.
            let woken = wake.notified();
            tokio::pin!(woken);
            woken.as_mut().enable();
            let Some(why) = busy() else {
                return Ok(());
            };
            let changed = self.inner.host.held.lock().insert(id.to_owned(), why.clone()) != Some(why.clone());
            if changed {
                self.emit(SwarmEvent::HostHeld {
                    id: id.to_owned(),
                    reason: Some(why),
                });
            }
            tokio::select! {
                () = &mut woken => {}
                () = tokio::time::sleep(HOST_POLL) => {}
                () = cancel.cancelled() => return Err(Error::Cancelled),
                () = self.inner.root_cancel.cancelled() => return Err(Error::Cancelled),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_values_are_read_in_bytes() {
        let text = "MemTotal:       16000000 kB\nMemFree:  100 kB\nMemAvailable:    2048 kB\n";
        assert_eq!(meminfo_bytes(text, "MemAvailable"), Some(2048 * 1024));
        assert_eq!(meminfo_bytes(text, "MemTotal"), Some(16_000_000 * 1024));
        assert_eq!(meminfo_bytes(text, "Mem"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_host_reads_and_defaults_are_sane() {
        let stats = host_stats().unwrap();
        assert!(stats.load >= 0.0 && stats.mem_available > 0);
        assert!(default_max_load() >= 2.0);
        assert!(default_min_free_mem() > 0 && default_min_free_mem() <= 8 << 30);
    }
}

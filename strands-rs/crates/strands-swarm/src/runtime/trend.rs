//! Disk trend: the guard samples free bytes and inodes per filesystem and fits a line to the last
//! day, so the roots hear that a disk is filling days before it is below the floor.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::Duration;

use super::disk::Space;

/// How far apart samples are kept.
pub const SAMPLE_EVERY: Duration = Duration::from_secs(10 * 60);

/// The window a line is fitted to.
const WINDOW: Duration = Duration::from_secs(24 * 3600);

/// The least span of samples worth fitting a line to.
const MIN_SPAN: Duration = Duration::from_secs(6 * 3600);

const DAY: f64 = 86_400.0;

/// What a fit says about one filesystem and metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// Bytes (or inodes) lost per day.
    pub per_day: f64,
    /// Days until the free amount reaches the floor at that rate.
    pub days: f64,
}

/// Free amounts over time, with whether the next short estimate may warn.
#[derive(Default)]
struct Series {
    /// `(time since the guard started, free)`, oldest first.
    samples: VecDeque<(Duration, u64)>,
    /// The floor at the last sample.
    floor: u64,
    /// Cleared by a warning, set again once the estimate is back above twice the threshold.
    spent: bool,
}

impl Series {
    fn push(&mut self, now: Duration, free: u64, floor: u64) {
        self.floor = floor;
        self.samples.push_back((now, free));
        while self
            .samples
            .front()
            .is_some_and(|(at, _)| now.saturating_sub(*at) > WINDOW)
        {
            self.samples.pop_front();
        }
    }

    /// A least-squares line through the samples, once they span [`MIN_SPAN`] and fall.
    fn estimate(&self) -> Option<Estimate> {
        let (first, last) = (self.samples.front()?, self.samples.back()?);
        if self.samples.len() < 3 || last.0 - first.0 < MIN_SPAN {
            return None;
        }
        let n = self.samples.len() as f64;
        let point =
            |(at, free): &(Duration, u64)| ((at.as_secs_f64() - first.0.as_secs_f64()) / DAY, *free as f64);
        let (sx, sy) = self
            .samples
            .iter()
            .map(point)
            .fold((0.0, 0.0), |a, (x, y)| (a.0 + x, a.1 + y));
        let (mx, my) = (sx / n, sy / n);
        let (mut sxy, mut sxx) = (0.0, 0.0);
        for (x, y) in self.samples.iter().map(point) {
            sxy += (x - mx) * (y - my);
            sxx += (x - mx) * (x - mx);
        }
        let per_day = -sxy / sxx;
        if per_day.is_nan() || per_day <= 0.0 {
            return None;
        }
        let left = last.1.saturating_sub(self.floor) as f64;
        Some(Estimate {
            per_day,
            days: left / per_day,
        })
    }
}

/// One filesystem: a path on it (for the warning's text) and its two series.
#[derive(Default)]
struct Filesystem {
    path: PathBuf,
    bytes: Series,
    inodes: Series,
}

/// A filesystem that will reach its floor within the threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Warning {
    pub path: PathBuf,
    /// Inodes rather than bytes.
    pub inodes: bool,
    pub estimate: Estimate,
}

/// The samples of every filesystem the guard watches, keyed by device.
pub struct Trends {
    warn_days: f64,
    last_sample: Option<Duration>,
    filesystems: HashMap<u64, Filesystem>,
}

impl Trends {
    /// Warn when a filesystem is `warn_days` or less from its floor; `0` never does.
    pub fn new(warn_days: u32) -> Self {
        Trends {
            warn_days: f64::from(warn_days),
            last_sample: None,
            filesystems: HashMap::new(),
        }
    }

    /// Take a sample of `readings` (path, space) if [`SAMPLE_EVERY`] has passed since the last,
    /// at `now`, and return the warnings it makes due. `floors` gives a filesystem's free-byte and
    /// free-inode floors (`0`: none, nothing to estimate against).
    pub fn observe(
        &mut self,
        now: Duration,
        readings: &[(PathBuf, Space)],
        floors: impl Fn(&Space) -> (u64, u64),
    ) -> Vec<Warning> {
        if self.warn_days <= 0.0 || self.last_sample.is_some_and(|at| now < at + SAMPLE_EVERY) {
            return Vec::new();
        }
        self.last_sample = Some(now);
        let mut seen: HashMap<u64, Space> = HashMap::new();
        for (path, space) in readings {
            let known = seen.entry(space.dev).or_insert(*space);
            known.bytes = known.bytes.min(space.bytes);
            known.inodes = known.inodes.min(space.inodes);
            let fs = self.filesystems.entry(space.dev).or_default();
            if fs.path.as_os_str().is_empty() || *path < fs.path {
                fs.path = path.clone();
            }
        }
        let warn_days = self.warn_days;
        let mut warnings = Vec::new();
        for (dev, space) in seen {
            let (bytes_floor, inodes_floor) = floors(&space);
            let fs = self.filesystems.get_mut(&dev).expect("inserted above");
            if bytes_floor > 0 {
                fs.bytes.push(now, space.bytes, bytes_floor);
            }
            if let Some((free, _)) = space.inodes.filter(|_| inodes_floor > 0) {
                fs.inodes.push(now, free, inodes_floor);
            }
            for (inodes, series) in [(false, &mut fs.bytes), (true, &mut fs.inodes)] {
                let estimate = series.estimate();
                let near = estimate.filter(|e| e.days < warn_days);
                if let (Some(estimate), false) = (near, series.spent) {
                    series.spent = true;
                    let path = fs.path.clone();
                    warnings.push(Warning {
                        path,
                        inodes,
                        estimate,
                    });
                } else if series.spent && estimate.is_none_or(|e| e.days > 2.0 * warn_days) {
                    series.spent = false;
                }
            }
        }
        warnings.sort_by(|a, b| a.path.cmp(&b.path).then(a.inodes.cmp(&b.inodes)));
        warnings
    }

    /// The bytes estimate of the filesystem with device `dev`, for `/status`.
    pub fn bytes_estimate(&self, dev: u64) -> Option<Estimate> {
        self.filesystems.get(&dev)?.bytes.estimate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TB: u64 = 1 << 40;

    fn space(bytes: u64) -> Vec<(PathBuf, Space)> {
        vec![(
            PathBuf::from("/x"),
            Space {
                dev: 1,
                bytes,
                inodes: None,
            },
        )]
    }

    /// Samples every 10 minutes for `hours`, free falling `per_day` from `start`; the warnings of
    /// each sample, in order.
    fn run(trends: &mut Trends, from: Duration, hours: u64, start: u64, per_day: i64) -> Vec<Warning> {
        let mut warned = Vec::new();
        for step in 0..=hours * 6 {
            let now = from + SAMPLE_EVERY * step as u32;
            let free = start as i64 - per_day * step as i64 / 144;
            warned.extend(trends.observe(now, &space(free as u64), |_| (20 << 30, 0)));
        }
        warned
    }

    #[test]
    fn a_steady_decline_under_the_threshold_warns_once_after_six_hours() {
        let early = run(&mut Trends::new(3), Duration::ZERO, 5, 2 * TB, TB as i64);
        assert_eq!(early, [], "fewer than 6 hours of samples");
        let warned = run(&mut Trends::new(3), Duration::ZERO, 36, 2 * TB, TB as i64);
        assert_eq!(warned.len(), 1, "{warned:?}");
        assert_eq!(warned[0].path, PathBuf::from("/x"));
        assert!(!warned[0].inodes);
        let Estimate { per_day, days } = warned[0].estimate;
        assert!((per_day / TB as f64 - 1.0).abs() < 0.01, "{per_day}");
        assert!(days < 3.0 && days > 1.0, "{days}");
    }

    #[test]
    fn a_decline_above_the_threshold_or_none_never_warns() {
        let mut trends = Trends::new(3);
        // 10 TB lasts ~10 days at 1 TB/day.
        assert_eq!(run(&mut trends, Duration::ZERO, 30, 10 * TB, TB as i64), []);
        assert!(trends.bytes_estimate(1).is_some_and(|e| e.days > 3.0));
        let mut steady = Trends::new(3);
        assert_eq!(run(&mut steady, Duration::ZERO, 30, TB, 0), []);
        assert_eq!(steady.bytes_estimate(1), None);
        let mut rising = Trends::new(3);
        assert_eq!(run(&mut rising, Duration::ZERO, 30, TB, -(TB as i64)), []);
        let mut off = Trends::new(0);
        assert_eq!(run(&mut off, Duration::ZERO, 30, TB / 4, TB as i64), []);
    }

    #[test]
    fn recovery_above_twice_the_threshold_rearms_it() {
        let mut trends = Trends::new(3);
        let hour = Duration::from_secs(3600);
        assert_eq!(run(&mut trends, Duration::ZERO, 10, 2 * TB, TB as i64).len(), 1);
        // Free space jumps back and then holds: the fit over the day flattens out.
        let steady = run(&mut trends, hour * 10 + SAMPLE_EVERY, 26, 20 * TB, 0);
        assert_eq!(steady, []);
        assert_eq!(trends.bytes_estimate(1), None);
        let again = run(&mut trends, hour * 37, 10, 3 * TB, TB as i64);
        assert_eq!(again.len(), 1, "{again:?}");
    }

    #[test]
    fn a_decline_that_stays_near_the_threshold_does_not_rearm() {
        let mut trends = Trends::new(3);
        assert_eq!(run(&mut trends, Duration::ZERO, 10, 2 * TB, TB as i64).len(), 1);
        // Slower, but still under twice the threshold (6 days): no second warning.
        let slower = run(
            &mut trends,
            Duration::from_secs(36000 + 600),
            40,
            4 * TB,
            TB as i64 / 2,
        );
        assert_eq!(slower, []);
    }

    #[test]
    fn samples_are_kept_ten_minutes_apart_and_inodes_warn_apart() {
        let mut trends = Trends::new(3);
        let reading = |free: u64| {
            vec![(
                PathBuf::from("/x"),
                Space {
                    dev: 7,
                    bytes: 50 * TB,
                    inodes: Some((free, 100_000_000)),
                },
            )]
        };
        let mut warned = Vec::new();
        for step in 0..=60u64 {
            let now = SAMPLE_EVERY * step as u32;
            // Called every minute, sampled every ten.
            for minute in 0..10 {
                let now = now + Duration::from_secs(60 * minute);
                let free = 50_000_000 - step * 500_000;
                warned.extend(trends.observe(now, &reading(free), |_| (20 << 30, 1_000_000)));
            }
        }
        assert_eq!(warned.len(), 1, "{warned:?}");
        assert!(warned[0].inodes);
        assert!((warned[0].estimate.per_day - 72e6).abs() < 1e6, "{:?}", warned[0]);
    }
}

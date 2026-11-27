//! Process limits the swarm raises for itself.

use nix::sys::resource::{getrlimit, setrlimit, Resource};

/// The open-files soft limit the swarm asks for, below the hard limit: enough for thousands of
/// agents' sessions, jobs and MCP servers, while children that still use `select()` (which
/// breaks past 1024 descriptors only when one actually opens that many) stay unaffected.
pub const OPEN_FILES: u64 = 65_536;

/// Raise the open-files soft limit to [`OPEN_FILES`], or the hard limit when that is lower.
/// The usual default of 1024 is reached by a large run on resume, when it reads every saved
/// conversation while its jobs and MCP servers hold descriptors; reads then fail with EMFILE
/// and the agent's history shows as unreadable. Returns the soft limit now in force.
pub fn raise_open_files() -> Option<u64> {
    let (soft, hard) = getrlimit(Resource::RLIMIT_NOFILE).ok()?;
    let want = OPEN_FILES.min(hard);
    if want > soft && setrlimit(Resource::RLIMIT_NOFILE, want, hard).is_ok() {
        return Some(want);
    }
    Some(soft)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_soft_limit_is_raised_towards_the_hard_one_and_never_lowered() {
        let (before, hard) = getrlimit(Resource::RLIMIT_NOFILE).unwrap();
        let now = raise_open_files().unwrap();
        assert!(now >= before, "{now} < {before}");
        assert!(now >= OPEN_FILES.min(hard) || now == before, "{now}");
        assert_eq!(getrlimit(Resource::RLIMIT_NOFILE).unwrap().0, now);
    }
}

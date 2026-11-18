//! Embed the git commit the binary is built from as `STRANDS_SWARM_COMMIT` (absent outside git),
//! and `STRANDS_SWARM_VERSION` for `--version`: the package version plus that short commit.

use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    (output.status.success() && !text.trim().is_empty()).then(|| text.trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let Some(commit) = git(&["rev-parse", "HEAD"]) else {
        println!("cargo:rustc-env=STRANDS_SWARM_VERSION={version}");
        return;
    };
    let short = commit.get(..9).unwrap_or(&commit);
    println!("cargo:rustc-env=STRANDS_SWARM_VERSION={version} ({short})");
    println!("cargo:rustc-env=STRANDS_SWARM_COMMIT={commit}");
    // Cargo reruns this script only when one of these files is newer than the script's last start,
    // judged from the fingerprint of the last run, which worktrees sharing a target dir also
    // share (their package hashes alike): a build in one can leave another embedding a stale
    // commit, and the reload self-test refuses such a binary (`version::stale`). Rerunning on
    // every build would recompile the crate every time. A build that sets
    // `STRANDS_SWARM_BUILD_HEAD` (e.g. to `$(git rev-parse HEAD)`) reruns this script whenever that
    // value differs from the last run's, which no shared fingerprint can hide.
    println!("cargo:rerun-if-env-changed=STRANDS_SWARM_BUILD_HEAD");
    // Rebuild when HEAD moves: its file, the branch it points at, packed refs, and HEAD's reflog,
    // which every commit, merge and checkout appends to (a branch ref packed when this ran has no
    // file to watch, and the loose one written by the next commit would go unnoticed).
    let mut refs = vec![
        "HEAD".to_owned(),
        "packed-refs".to_owned(),
        "logs/HEAD".to_owned(),
    ];
    refs.extend(git(&["symbolic-ref", "-q", "HEAD"]));
    for name in refs {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", &name]) {
            if Path::new(&path).exists() {
                println!("cargo:rerun-if-changed={path}");
            }
        }
    }
}

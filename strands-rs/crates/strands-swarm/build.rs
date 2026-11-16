//! Embed the git commit the binary is built from as `STRANDS_SWARM_COMMIT` (absent outside git).

use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    (output.status.success() && !text.trim().is_empty()).then(|| text.trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let Some(commit) = git(&["rev-parse", "HEAD"]) else {
        return;
    };
    println!("cargo:rustc-env=STRANDS_SWARM_COMMIT={commit}");
    // Rebuild when HEAD moves: its file, the branch it points at, and packed refs.
    let mut refs = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
    refs.extend(git(&["symbolic-ref", "-q", "HEAD"]));
    for name in refs {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", &name]) {
            if Path::new(&path).exists() {
                println!("cargo:rerun-if-changed={path}");
            }
        }
    }
}

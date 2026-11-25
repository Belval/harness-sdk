//! The signals that end the swarm from outside, named so that the stop can say why.

use tokio::signal::unix::{signal, Signal, SignalKind};

/// SIGTERM and SIGHUP (a closed terminal or a killed tmux session): the swarm stops cleanly on
/// them, saving its state, and exits with the status a death by that signal has (`128 + n`).
pub struct Terminate {
    streams: Option<(Signal, Signal)>,
}

impl Terminate {
    /// Start catching the signals; call it before the work they interrupt.
    pub fn new() -> Self {
        let streams = signal(SignalKind::terminate())
            .and_then(|term| signal(SignalKind::hangup()).map(|hup| (term, hup)))
            .ok();
        Self { streams }
    }

    /// The name of the next signal caught; never ready when they cannot be caught.
    pub async fn next(&mut self) -> &'static str {
        let Some((term, hup)) = &mut self.streams else {
            return std::future::pending().await;
        };
        tokio::select! {
            _ = term.recv() => "SIGTERM",
            _ = hup.recv() => "SIGHUP",
        }
    }
}

impl Default for Terminate {
    fn default() -> Self {
        Self::new()
    }
}

/// The exit status of a process killed by the signal `name` ([`Terminate::next`]).
pub fn exit_code(name: &str) -> u8 {
    128 + match name {
        "SIGHUP" => 1,
        _ => 15,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_follow_the_shell_convention() {
        assert_eq!(exit_code("SIGTERM"), 143);
        assert_eq!(exit_code("SIGHUP"), 129);
    }

    #[tokio::test]
    async fn a_caught_signal_is_named() {
        let mut caught = Terminate::new();
        nix::sys::signal::raise(nix::sys::signal::Signal::SIGHUP).unwrap();
        assert_eq!(caught.next().await, "SIGHUP");
    }
}

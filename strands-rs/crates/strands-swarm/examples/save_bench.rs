//! Time and peak heap use of one conversation save and one `swarm.json` save.
//!
//! `cargo run --release -p strands-swarm --example save_bench [-- MB...]` (default `1 12`).
//! `SAVE_BENCH_SNAPSHOT=<snapshot_latest.json>` cycles a real conversation's messages up to each
//! size instead of synthetic ones; `SAVE_BENCH_STATE=<swarm.json>` saves a real state.

use std::time::{Duration, Instant};

use peak_alloc::PeakAlloc;
use serde_json::{json, Value};
use strands_agents::testing::MockMessageModel;
use strands_agents::types::Message;
use strands_agents::Agent;
use strands_swarm::state::{Persister, SwarmState};

#[global_allocator]
static HEAP: PeakAlloc = PeakAlloc;

const MB: usize = 1 << 20;

/// Words of varied length, with the quotes and newlines real tool output has.
fn text(seed: &mut u64, len: usize) -> String {
    const WORDS: [&str; 12] = [
        "fn",
        "let",
        "self.inner",
        "\"path\"",
        "error:",
        "Ok(())",
        "\n",
        "  ",
        "cargo",
        "test",
        "{",
        "}\n",
    ];
    let mut out = String::with_capacity(len + 16);
    while out.len() < len {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push_str(WORDS[(*seed >> 33) as usize % WORDS.len()]);
        out.push(' ');
    }
    out
}

/// A synthetic conversation of tool calls and results, about `bytes` of JSON.
fn synthetic(bytes: usize) -> Vec<Message> {
    let mut seed = 7;
    let first = json!({"role": "user", "content": [{"text": text(&mut seed, 2000)}]});
    let mut messages = vec![serde_json::from_value(first).unwrap()];
    let mut size = 0;
    let mut i = 0;
    while size < bytes {
        i += 1;
        let id = format!("tooluse_{i:020}");
        let call = json!({"role": "assistant", "content": [
            {"text": text(&mut seed, 300)},
            {"toolUse": {"toolUseId": id, "name": "shell",
                         "input": {"command": text(&mut seed, 120)}}}]});
        let result = json!({"role": "user", "content": [
            {"toolResult": {"toolUseId": id, "status": "success",
                            "content": [{"text": text(&mut seed, 3000)}]}}]});
        size += call.to_string().len() + result.to_string().len();
        messages.push(serde_json::from_value(call).unwrap());
        messages.push(serde_json::from_value(result).unwrap());
    }
    messages
}

/// `real`'s messages repeated up to about `bytes` of JSON.
fn cycled(real: &[Message], bytes: usize) -> Vec<Message> {
    let mut messages = Vec::new();
    let mut size = 0;
    for message in real.iter().cycle() {
        if size >= bytes {
            break;
        }
        size += serde_json::to_vec(message).unwrap().len();
        messages.push(message.clone());
    }
    messages
}

/// Median time and largest heap growth of the measured spans.
#[derive(Default)]
struct Probe {
    times: Vec<Duration>,
    peak: usize,
    base: usize,
    start: Option<Instant>,
}

impl Probe {
    fn start(&mut self) {
        self.base = HEAP.current_usage();
        HEAP.reset_peak_usage();
        self.start = Some(Instant::now());
    }

    fn stop(&mut self) {
        self.times.push(self.start.take().unwrap().elapsed());
        self.peak = self.peak.max(HEAP.peak_usage().saturating_sub(self.base));
    }

    fn report(mut self, what: &str, file: &std::path::Path) {
        self.times.sort();
        let time = self.times[self.times.len() / 2];
        let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
        println!(
            "{what:<34} {:>8.1} ms  peak +{:>6.1} MB  file {:>6.2} MB",
            time.as_secs_f64() * 1e3,
            self.peak as f64 / MB as f64,
            size as f64 / MB as f64
        );
    }
}

#[tokio::main]
async fn main() {
    let dir = tempfile::tempdir().unwrap();
    let sessions = dir.path().join("sessions");
    let real: Option<Vec<Message>> = std::env::var_os("SAVE_BENCH_SNAPSHOT").map(|path| {
        let mut snapshot: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        serde_json::from_value(snapshot["data"]["messages"].take()).unwrap()
    });
    let sizes: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let sizes = if sizes.is_empty() { vec![1, 12] } else { sizes };
    for mb in sizes {
        let id = format!("a{mb}");
        let messages = match &real {
            Some(real) => cycled(real, mb * MB),
            None => synthetic(mb * MB),
        };
        let agent = Agent::builder()
            .model(MockMessageModel::new(Vec::new()))
            .messages(messages)
            .build()
            .unwrap();
        let manager = strands_swarm::agent_session_manager(&sessions, &id).unwrap();
        let file = sessions.join(format!(
            "{id}/scopes/agent/{}/snapshots/snapshot_latest.json",
            agent.id()
        ));
        let runs = if mb >= 8 { 5 } else { 15 };
        let mut changed = Probe::default();
        for n in 0..runs {
            agent.with_messages(|m| m.push(Message::user(format!("mail {n}"))));
            changed.start();
            manager.save_snapshot(&agent, true).await.unwrap();
            changed.stop();
        }
        changed.report(&format!("conversation {mb} MB, changed"), &file);
        let mut same = Probe::default();
        for _ in 0..runs {
            same.start();
            manager.save_snapshot(&agent, true).await.unwrap();
            same.stop();
        }
        same.report(&format!("conversation {mb} MB, unchanged"), &file);
    }

    let mut state: SwarmState = match std::env::var_os("SAVE_BENCH_STATE") {
        Some(path) => serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap(),
        None => {
            let mut seed = 11;
            let agents: Vec<Value> = (1..=330)
                .map(|i| {
                    json!({"id": format!("a{i}"), "parent": "a1", "cwd": "/tmp/work",
                           "task": text(&mut seed, 7000), "status": "idle", "mailbox": []})
                })
                .collect();
            let state = json!({"run_id": "bench", "cwd": "/tmp", "next_id": 331, "agents": agents});
            serde_json::from_value(state).unwrap()
        }
    };
    let persister = Persister::new(dir.path());
    let file = dir.path().join("swarm.json");
    let mut generation = 0;
    let mut changed = Probe::default();
    for _ in 0..15 {
        generation += 1;
        state.next_id += 1;
        changed.start();
        persister.write(generation, &state).await.unwrap();
        changed.stop();
    }
    changed.report("swarm.json, changed", &file);
    let mut same = Probe::default();
    for _ in 0..15 {
        generation += 1;
        same.start();
        persister.write(generation, &state).await.unwrap();
        same.stop();
    }
    same.report("swarm.json, unchanged", &file);
}

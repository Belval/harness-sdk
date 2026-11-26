//! Advisory path claims ([`crate::claims`]): the note an `edit`/`write` result carries when another
//! live agent is changing the same file of the same repo, and the `editing:` part of `list_agents`.

use std::collections::{BTreeSet, HashMap};

use super::*;
use crate::claims::{self, Key, Repo};

/// A live agent and the files it claims.
struct Claimant {
    id: String,
    label: String,
    claimed: BTreeSet<Key>,
}

impl Swarm {
    /// Append a note to the `edit`/`write` results that touch a file another live agent claims.
    pub(super) fn install_claims_hook(&self, agent: &Agent, id: &str) {
        let swarm = self.clone();
        let id = id.to_owned();
        agent
            .hooks()
            .add_callback_boxed(move |event: &mut AfterToolCallEvent| {
                let (swarm, id) = (swarm.clone(), id.clone());
                Box::pin(async move {
                    let edits = matches!(event.tool_use.name.as_str(), "edit" | "write");
                    let path = event.tool_use.input.get("path").and_then(|p| p.as_str());
                    let Some(path) =
                        path.filter(|_| edits && event.result.status == ToolResultStatus::Success)
                    else {
                        return Ok(());
                    };
                    for note in swarm.claim_notes(&id, path).await {
                        event
                            .result
                            .content
                            .push(ToolResultContent::Text(format!("\n\n{note}")));
                    }
                    Ok(())
                })
            });
    }

    /// Record that `id` changed `path`, and say which other live agents claim it, once per turn.
    async fn claim_notes(&self, id: &str, path: &str) -> Vec<String> {
        let Some(node) = self.node(id) else {
            return Vec::new();
        };
        let Some((key, top)) = claims::key_of(&node.workdir.get().join(path)) else {
            return Vec::new();
        };
        self.inner
            .claims
            .lock()
            .agents
            .entry(id.to_owned())
            .or_default()
            .edited
            .insert(key.clone(), (Instant::now(), top));
        let others = self.claimants(Some(&key.0)).await;
        let mut claims = self.inner.claims.lock();
        let mine = claims.agents.entry(id.to_owned()).or_default();
        let sharing = others.iter().filter(|o| o.id != id && o.claimed.contains(&key));
        let fresh = sharing.filter(|o| mine.noted.insert((key.clone(), o.id.clone())));
        fresh.map(|other| claims::note(&key.1, &other.label)).collect()
    }

    /// Per live agent with claims: `, editing: 2 files; shares a.rs with b (a12)`, for
    /// `list_agents`.
    pub async fn claims_report(&self) -> HashMap<String, String> {
        let claimants = self.claimants(None).await;
        let mut report = HashMap::new();
        for mine in claimants.iter().filter(|c| !c.claimed.is_empty()) {
            let others: Vec<(String, BTreeSet<Key>)> = claimants
                .iter()
                .filter(|o| o.id != mine.id)
                .map(|o| (o.label.clone(), o.claimed.clone()))
                .collect();
            let mut parts = vec![claims::editing(mine.claimed.len())];
            parts.extend(claims::shared_with(&mine.claimed, &others));
            report.insert(mine.id.clone(), format!(", {}", parts.join("; ")));
        }
        report
    }

    /// The live agents working in a git checkout (of the repo `common`, when given) with the files
    /// they claim: those they changed this turn, and the uncommitted ones of a checkout they have
    /// to themselves. Runs `git status` for the checkouts whose last one is older than
    /// [`claims::TTL`].
    async fn claimants(&self, common: Option<&Path>) -> Vec<Claimant> {
        let live: Vec<(Arc<Node>, Repo)> = self
            .nodes()
            .into_iter()
            .filter(|node| node.status.lock().is_live())
            .filter_map(|node| claims::repo_at(&node.workdir.get()).map(|repo| (node, repo)))
            .filter(|(_, repo)| common.is_none_or(|common| repo.common == common))
            .collect();
        let mut repos: Vec<&Repo> = live.iter().map(|(_, repo)| repo).collect();
        repos.sort_by(|a, b| a.top.cmp(&b.top));
        repos.dedup();
        futures::future::join_all(repos.into_iter().map(|repo| self.refresh_status(repo))).await;
        let claims = self.inner.claims.lock();
        let alone = |top: &Path| live.iter().filter(|(_, repo)| repo.top == top).count() == 1;
        live.iter()
            .map(|(node, repo)| {
                let mut claimed: BTreeSet<Key> = claims
                    .agents
                    .get(&node.id)
                    .map(|mine| mine.edited.keys().cloned().collect())
                    .unwrap_or_default();
                if let Some(dirty) = claims.status.get(&repo.top).filter(|_| alone(&repo.top)) {
                    claimed.extend(dirty.paths.iter().map(|path| (repo.common.clone(), path.clone())));
                }
                Claimant {
                    id: node.id.clone(),
                    label: node.label(),
                    claimed,
                }
            })
            .collect()
    }

    /// Take the checkout's `git status` unless one is younger than [`claims::TTL`].
    async fn refresh_status(&self, repo: &Repo) {
        let fresh = self
            .inner
            .claims
            .lock()
            .status
            .get(&repo.top)
            .map(|d| d.at.elapsed() < claims::TTL);
        if fresh == Some(true) {
            return;
        }
        if let Some(dirty) = claims::dirty(repo).await {
            self.inner.claims.lock().settle(repo.top.clone(), dirty);
        }
    }

    /// A turn starts (notes are given again) or the agent is cancelled (its claims go). What an
    /// idle agent changed stays claimed while it is uncommitted ([`claims::Claims::settle`]).
    pub(super) fn claims_turn(&self, id: &str, status: &AgentStatus) {
        let mut claims = self.inner.claims.lock();
        match status {
            AgentStatus::Running => claims
                .agents
                .get_mut(id)
                .iter_mut()
                .for_each(|mine| mine.noted.clear()),
            AgentStatus::Cancelled => drop(claims.agents.remove(id)),
            _ => {}
        }
    }
}

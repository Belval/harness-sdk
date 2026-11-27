//! The `land` primitive: an agent asks for its commits to be rebased, checked and fast-forwarded
//! onto a branch it does not own ([`crate::land`] does the git work in a background job).

use super::*;

impl Swarm {
    /// Whether agents get the `land` tool ([`SwarmConfig::allow_land`]).
    pub fn allows_land(&self) -> bool {
        self.inner.config.allow_land
    }

    /// Start a background job that lands `owner`'s commits ([`LandRequest`]) on another branch:
    /// rebased onto it in a scratch worktree, checked there with a private `CARGO_TARGET_DIR`,
    /// then fast-forwarded. The owner hears the outcome as a `job` message; on a conflict or a
    /// failing check nothing is changed. Fails at once when landing is off or a branch is unknown.
    pub async fn land(&self, owner: &str, request: LandRequest) -> Result<Job> {
        if !self.allows_land() {
            return Err(Error::other(
                "landing is off; the swarm must be started with --allow-land",
            ));
        }
        let node = self
            .node(owner)
            .ok_or_else(|| Error::other(format!("no agent {owner}")))?;
        let own = self.own_branches(&node).await;
        let label = match request.branch {
            Some(branch) if own.contains(&branch) => branch,
            Some(branch) => {
                return Err(Error::other(format!(
                    "{branch} is not yours to land: only your own branch or a branch of an agent \
                     you spawned (directly or not) can be landed ({})",
                    listing(&own)
                )))
            }
            None => own.first().cloned().ok_or_else(|| {
                Error::other(
                    "you have no branch of your own (spawn agents with `worktree: true`); name \
                     a branch of an agent you spawned with `branch`",
                )
            })?,
        };
        let source = worktree::git(
            &node.cwd,
            &["rev-parse", "--verify", &format!("refs/heads/{label}^{{commit}}")],
        )
        .await
        .map_err(|_| Error::other(format!("unknown branch {label}")))?;
        let root = self.root_of(&node);
        let default_into = crate::status::current_branch(&root.cwd).await;
        let into = match (request.into, &default_into) {
            (Some(into), _) => into,
            (None, Some(default)) => default.clone(),
            (None, None) => {
                return Err(Error::other(format!(
                    "{} has no branch checked out; name the target with `into`",
                    root.label()
                )))
            }
        };
        let allowed = &self.inner.config.land_into;
        if default_into.as_ref() != Some(&into)
            && !allowed.iter().any(|glob| crate::land::glob_match(glob, &into))
        {
            let default = match &default_into {
                Some(default) => format!("only {default} (the branch {} has checked out)", root.label()),
                None => "no branch by default".to_owned(),
            };
            let extra = match allowed.is_empty() {
                true => "the user allows more with --land-into GLOB".to_owned(),
                false => format!("or one matching {}", allowed.join(", ")),
            };
            return Err(Error::other(format!(
                "landing on {into} is not allowed: agents may land on {default}; {extra}"
            )));
        }
        if into == label {
            return Err(Error::other(format!("{label} cannot be landed on itself")));
        }
        let dir = jobs_dir(&self.inner.config).join("land");
        let stamp = Utc::now().format("%Y%m%d-%H%M%S%3f");
        let scratch = dir.join(format!("{owner}-{stamp}"));
        let target_dir = land_target_dir(&self.inner.config);
        let plan = crate::land::LandPlan {
            repo: node.cwd.clone(),
            source,
            label: label.clone(),
            into: into.clone(),
            gates: self.inner.config.gates.clone(),
            commit_rules: (self.inner.config.commit_hooks.as_ref())
                .map(|hooks| hooks.rules.clone())
                .unwrap_or_default(),
            checks: request.check.unwrap_or_default(),
            scratch: scratch.clone(),
            target_dir,
        };
        let plan_path = scratch.with_extension("json");
        tokio::fs::create_dir_all(&dir).await?;
        let json = serde_json::to_vec_pretty(&plan).map_err(Error::other)?;
        tokio::fs::write(&plan_path, json).await?;
        let exe = match &self.inner.config.exe {
            Some(exe) => exe.clone(),
            None => std::env::current_exe()?,
        };
        let command = format!(
            "{}{}{}",
            crate::land::sh_quote(&exe.to_string_lossy()),
            crate::land::LAND_FLAG,
            crate::land::sh_quote(&plan_path.to_string_lossy())
        );
        // Its checks build into a target dir of their own: not while space is low.
        self.wait_for_host(owner, true, &node.turn_token()).await?;
        self.start_job(owner, &command, Some(format!("land {label} → {into}")), false)
            .await
    }

    /// [`crate::land::tidy_when_idle`]; with `target`, the checks' shared target dir goes too.
    pub(super) async fn tidy_landings(&self, target: bool) -> Option<Collected> {
        let target = target.then(|| land_target_dir(&self.inner.config));
        crate::land::tidy_when_idle(&self.inner.jobs, target.as_deref()).await
    }

    /// Branches `node` may land: its own (a root agent's is the one it has checked out), then
    /// those of the agents below it.
    async fn own_branches(&self, node: &Arc<Node>) -> Vec<String> {
        let mut own: Vec<String> = self
            .subtree(&node.id)
            .iter()
            .filter_map(|node| node.branch.clone())
            .collect();
        if node.branch.is_none() && node.parent == USER {
            if let Some(branch) = crate::status::current_branch(&node.cwd).await {
                own.insert(0, branch);
            }
        }
        let mut seen = std::collections::HashSet::new();
        own.retain(|branch| seen.insert(branch.clone()));
        own
    }

    /// The top-level agent above `node` (the one the user started).
    pub(super) fn root_of(&self, node: &Arc<Node>) -> Arc<Node> {
        let mut current = node.clone();
        while let Some(parent) = self.node(&current.parent) {
            current = parent;
        }
        current
    }
}

/// [`crate::land::target_dir`] of this swarm.
pub(super) fn land_target_dir(config: &SwarmConfig) -> PathBuf {
    let land = jobs_dir(config).join("land");
    crate::land::target_dir(config.cargo_target_dir.as_deref(), &config.run_id, &land)
}

fn listing(branches: &[String]) -> String {
    match branches.is_empty() {
        true => "you have none".to_owned(),
        false => format!("yours: {}", branches.join(", ")),
    }
}

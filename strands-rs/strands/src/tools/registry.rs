//! Name-keyed registry of tools. Ports `registry/tool-registry.ts`.

use std::sync::Arc;

use crate::errors::StrandsError;
use crate::tools::Tool;
use crate::types::tools::ToolSpec;

const TOOL_NAME_MAX_LENGTH: usize = 64;

/// Registry for managing [`Tool`] instances with name-based operations.
///
/// Insertion order is preserved (a `Vec` of `(name, tool)`), matching the
/// TypeScript SDK's `Map`, so `tool_specs()` reports tools in registration order.
///
/// The registry has two layers, mirroring the Python SDK's warm-registry pattern:
/// a **base** set built at construction (shared read-only behind an `Arc`, so a
/// [`ToolRegistry::fork`] is cheap) and a per-instance **dynamic** set added at
/// runtime via [`ToolRegistry::register_dynamic_tool`]. A dynamic tool overrides
/// a base tool of the same name; resolution and specs see the union.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    base: Arc<Vec<(String, Arc<dyn Tool>)>>,
    dynamic: Vec<(String, Arc<dyn Tool>)>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        ToolRegistry::default()
    }

    /// Shares this registry's base layer with a fresh, empty dynamic layer.
    ///
    /// The expensive base set is shared read-only (an `Arc` clone); the returned
    /// registry gets its own dynamic layer, so dynamic tools it registers do not
    /// leak back. Ports the Python thread fast path (`cloned.registry =
    /// prebuilt.registry`, empty `dynamic_tools`).
    pub fn fork(&self) -> Self {
        ToolRegistry {
            base: Arc::clone(&self.base),
            dynamic: Vec::new(),
        }
    }

    /// Registers a tool in the base layer.
    ///
    /// # Errors
    /// Returns [`StrandsError::ToolValidation`] if the name is invalid, already
    /// registered in the base, or conflicts with an existing base name that
    /// differs only by `-`/`_`, mirroring the TypeScript `add` validation.
    pub fn add(&mut self, tool: Arc<dyn Tool>) -> Result<(), StrandsError> {
        let name = tool.name().to_string();
        validate_tool(&name, tool.as_ref())?;
        if base_get(&self.base, &name).is_some() {
            return Err(StrandsError::ToolValidation(format!(
                "Tool with name '{name}' already registered"
            )));
        }
        check_normalized_conflict(
            self.base.iter().map(|(existing, _)| existing.as_str()),
            &name,
        )?;
        // Copy-on-write: cheap while the base is unshared (build time), which is
        // when `add` is used; a base shared via `fork` copies once here.
        Arc::make_mut(&mut self.base).push((name, tool));
        Ok(())
    }

    /// Registers a runtime **dynamic** tool. A dynamic tool overrides a base tool
    /// of the same name, and replaces any prior dynamic tool of that name. Ports
    /// `register_dynamic_tool`.
    ///
    /// # Errors
    /// Returns [`StrandsError::ToolValidation`] if the name or description is invalid.
    pub fn register_dynamic_tool(&mut self, tool: Arc<dyn Tool>) -> Result<(), StrandsError> {
        let name = tool.name().to_string();
        validate_tool(&name, tool.as_ref())?;
        self.dynamic.retain(|(existing, _)| existing != &name);
        self.dynamic.push((name, tool));
        Ok(())
    }

    /// Retrieves a tool by exact name, preferring a dynamic tool over a base one.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        dynamic_get(&self.dynamic, name).or_else(|| base_get(&self.base, name))
    }

    /// Resolves a tool name using the TypeScript resolution order: exact match,
    /// then underscore-to-hyphen substitution, then case-insensitive match.
    /// Dynamic tools take precedence over base tools at each step.
    ///
    /// # Errors
    /// Returns [`StrandsError::ToolNotFound`] when no tool matches.
    pub fn resolve(&self, name: &str) -> Result<&Arc<dyn Tool>, StrandsError> {
        if let Some(tool) = self.get(name) {
            return Ok(tool);
        }
        if name.contains('_') {
            if let Some(tool) = self.find(|key| key.replace('-', "_") == name) {
                return Ok(tool);
            }
        }
        let lower = name.to_lowercase();
        if let Some(tool) = self.find(|key| key.to_lowercase() == lower) {
            return Ok(tool);
        }
        Err(StrandsError::ToolNotFound(name.to_string()))
    }

    /// Removes a tool by name from both the base and dynamic layers. No-op if
    /// absent.
    pub fn remove(&mut self, name: &str) {
        if base_get(&self.base, name).is_some() {
            Arc::make_mut(&mut self.base).retain(|(key, _)| key != name);
        }
        self.dynamic.retain(|(key, _)| key != name);
    }

    /// Returns all tools (base + dynamic union, dynamic overriding base by name),
    /// base tools first in registration order, then dynamic tools.
    pub fn list(&self) -> Vec<Arc<dyn Tool>> {
        self.combined().map(|(_, tool)| Arc::clone(tool)).collect()
    }

    /// The runtime dynamic tools, in registration order. Ports `dynamic_tools`.
    pub fn dynamic_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.dynamic
            .iter()
            .map(|(_, tool)| Arc::clone(tool))
            .collect()
    }

    /// Returns the specs of all tools (base + dynamic union). This is what the
    /// agent offers the model, so dynamic tools are included.
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.combined().map(|(_, tool)| tool.tool_spec()).collect()
    }

    /// The union of base and dynamic tool specs — the progressive-disclosure
    /// surface. Ports `get_all_tool_specs`; today it is the full union (an alias
    /// of [`ToolRegistry::tool_specs`]). The Python spec-filtering monkey-patch
    /// (live `hidden_tools` for skill activation) is the skills subsystem's
    /// concern and is deferred.
    pub fn get_all_tool_specs(&self) -> Vec<ToolSpec> {
        self.tool_specs()
    }

    /// Iterates base tools not shadowed by a dynamic tool, then dynamic tools.
    fn combined(&self) -> impl Iterator<Item = (&String, &Arc<dyn Tool>)> {
        let base = self
            .base
            .iter()
            .filter(|(name, _)| dynamic_get(&self.dynamic, name).is_none())
            .map(|(name, tool)| (name, tool));
        base.chain(self.dynamic.iter().map(|(name, tool)| (name, tool)))
    }

    /// Finds the first tool (dynamic before base) whose name matches `predicate`.
    fn find(&self, predicate: impl Fn(&str) -> bool) -> Option<&Arc<dyn Tool>> {
        self.dynamic
            .iter()
            .chain(self.base.iter())
            .find(|(key, _)| predicate(key))
            .map(|(_, tool)| tool)
    }
}

fn base_get<'a>(base: &'a [(String, Arc<dyn Tool>)], name: &str) -> Option<&'a Arc<dyn Tool>> {
    base.iter()
        .find(|(key, _)| key == name)
        .map(|(_, tool)| tool)
}

fn dynamic_get<'a>(
    dynamic: &'a [(String, Arc<dyn Tool>)],
    name: &str,
) -> Option<&'a Arc<dyn Tool>> {
    dynamic
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, tool)| tool)
}

fn validate_tool(name: &str, tool: &dyn Tool) -> Result<(), StrandsError> {
    validate_name(name)?;
    if tool.description().is_empty() {
        return Err(StrandsError::ToolValidation(
            "Tool description must be a non-empty string".to_string(),
        ));
    }
    Ok(())
}

fn check_normalized_conflict<'a>(
    existing_names: impl Iterator<Item = &'a str>,
    name: &str,
) -> Result<(), StrandsError> {
    let normalized = name.replace('-', "_");
    for existing in existing_names {
        if existing != name && existing.replace('-', "_") == normalized {
            return Err(StrandsError::ToolValidation(format!(
                "Tool name '{name}' already exists as '{existing}'. \
                 Cannot add a duplicate tool which differs by a '-' or '_'"
            )));
        }
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), StrandsError> {
    if name.is_empty() || name.len() > TOOL_NAME_MAX_LENGTH {
        return Err(StrandsError::ToolValidation(
            "Tool name must be between 1 and 64 characters".to_string(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(StrandsError::ToolValidation(
            "Tool name must contain only alphanumeric characters, hyphens, and underscores"
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Ports `registry/__tests__/tool-registry.test.ts`. Rust has no `-`/`_`- vs
    //! case-only tension in the mock, so the same named tools are used.

    use super::*;
    use crate::tools::{Tool, ToolContext};
    use crate::types::tools::ToolSpec;
    use async_trait::async_trait;

    struct MockTool {
        name: String,
        description: String,
    }

    impl MockTool {
        fn arc(name: &str) -> Arc<dyn Tool> {
            Arc::new(MockTool {
                name: name.to_string(),
                description: "A valid tool description.".to_string(),
            })
        }

        fn arc_with_description(name: &str, description: &str) -> Arc<dyn Tool> {
            Arc::new(MockTool {
                name: name.to_string(),
                description: description.to_string(),
            })
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            &self.description
        }
        fn tool_spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.clone(),
                description: self.description.clone(),
                input_schema: None,
                output_schema: None,
            }
        }
        async fn invoke(&self, _context: ToolContext) -> Result<serde_json::Value, StrandsError> {
            Ok(serde_json::Value::Null)
        }
    }

    // add: registers a single tool
    #[test]
    fn registers_a_single_tool() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("valid-tool")).unwrap();
        assert_eq!(registry.list().len(), 1);
        assert!(registry.get("valid-tool").is_some());
    }

    // add: registers multiple tools in order
    #[test]
    fn registers_tools_in_registration_order() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("tool-1")).unwrap();
        registry.add(MockTool::arc("tool-2")).unwrap();
        let names: Vec<_> = registry
            .list()
            .iter()
            .map(|tool| tool.name().to_string())
            .collect();
        assert_eq!(names, vec!["tool-1", "tool-2"]);
    }

    // add: throws ToolValidationError for a duplicate tool name
    #[test]
    fn rejects_duplicate_tool_name() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("duplicate")).unwrap();
        let error = registry.add(MockTool::arc("duplicate")).unwrap_err();
        assert!(matches!(error, StrandsError::ToolValidation(_)));
        assert_eq!(
            error.to_string(),
            "Tool with name 'duplicate' already registered"
        );
    }

    // add: throws when a name differs only by '-' vs '_'
    #[test]
    fn rejects_name_differing_only_by_hyphen_underscore() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("foo-bar")).unwrap();
        let error = registry.add(MockTool::arc("foo_bar")).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Tool name 'foo_bar' already exists as 'foo-bar'. \
             Cannot add a duplicate tool which differs by a '-' or '_'"
        );
    }

    // add: throws ToolValidationError for an invalid tool name pattern
    #[test]
    fn rejects_invalid_name_pattern() {
        let mut registry = ToolRegistry::new();
        let error = registry.add(MockTool::arc("invalid name!")).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Tool name must contain only alphanumeric characters, hyphens, and underscores"
        );
    }

    // add: throws for a name that is too long / too short
    #[test]
    fn rejects_name_length_out_of_bounds() {
        let mut registry = ToolRegistry::new();
        let long_name = "a".repeat(65);
        let long_error = registry.add(MockTool::arc(&long_name)).unwrap_err();
        assert_eq!(
            long_error.to_string(),
            "Tool name must be between 1 and 64 characters"
        );
        let short_error = registry.add(MockTool::arc("")).unwrap_err();
        assert_eq!(
            short_error.to_string(),
            "Tool name must be between 1 and 64 characters"
        );
    }

    // add: throws ToolValidationError for an empty string description
    #[test]
    fn rejects_empty_description() {
        let mut registry = ToolRegistry::new();
        let error = registry
            .add(MockTool::arc_with_description("tool-1", ""))
            .unwrap_err();
        assert!(matches!(error, StrandsError::ToolValidation(_)));
        assert_eq!(
            error.to_string(),
            "Tool description must be a non-empty string"
        );
    }

    // add: registers a tool with a name at the maximum length
    #[test]
    fn accepts_name_at_maximum_length() {
        let mut registry = ToolRegistry::new();
        let name = "a".repeat(64);
        assert!(registry.add(MockTool::arc(&name)).is_ok());
    }

    // get: returns None for a non-existent tool
    #[test]
    fn get_returns_none_for_missing_tool() {
        let registry = ToolRegistry::new();
        assert!(registry.get("non-existent").is_none());
    }

    // resolve: exact name match
    #[test]
    fn resolve_exact_match() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("my-tool")).unwrap();
        assert_eq!(registry.resolve("my-tool").unwrap().name(), "my-tool");
    }

    // resolve: underscore-to-hyphen substitution
    #[test]
    fn resolve_underscore_to_hyphen() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("my-tool")).unwrap();
        assert_eq!(registry.resolve("my_tool").unwrap().name(), "my-tool");
    }

    // resolve: case-insensitive match
    #[test]
    fn resolve_case_insensitive() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("MyTool")).unwrap();
        assert_eq!(registry.resolve("mytool").unwrap().name(), "MyTool");
    }

    // resolve: prefers exact match over case-insensitive match
    #[test]
    fn resolve_prefers_exact_over_case_insensitive() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("mytool")).unwrap();
        registry.add(MockTool::arc("MYTOOL")).unwrap();
        assert_eq!(registry.resolve("mytool").unwrap().name(), "mytool");
    }

    // resolve: throws ToolNotFoundError when no tool matches (and message form)
    #[test]
    fn resolve_missing_tool_errors_with_name() {
        let registry = ToolRegistry::new();
        let error = match registry.resolve("missing") {
            Ok(_) => panic!("expected resolve() to error"),
            Err(error) => error,
        };
        assert!(matches!(error, StrandsError::ToolNotFound(ref name) if name == "missing"));
        assert_eq!(error.to_string(), "Tool 'missing' not found");
    }

    // remove: removes a tool; no-op for a non-existent tool
    #[test]
    fn remove_tool() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("remove-me")).unwrap();
        registry.remove("remove-me");
        assert!(registry.get("remove-me").is_none());
        registry.remove("non-existent"); // no panic
    }

    // list: empty by default
    #[test]
    fn list_empty_by_default() {
        let registry = ToolRegistry::new();
        assert!(registry.list().is_empty());
    }

    // register_dynamic_tool: resolvable and included in the union specs
    #[test]
    fn dynamic_tool_resolves_and_appears_in_specs() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("base-tool")).unwrap();
        registry
            .register_dynamic_tool(MockTool::arc("dynamic-tool"))
            .unwrap();

        assert!(registry.get("dynamic-tool").is_some());
        assert_eq!(
            registry.resolve("dynamic_tool").unwrap().name(),
            "dynamic-tool"
        );
        let names: Vec<_> = registry
            .tool_specs()
            .iter()
            .map(|spec| spec.name.clone())
            .collect();
        assert_eq!(names, vec!["base-tool", "dynamic-tool"]);
        assert_eq!(registry.dynamic_tools().len(), 1);
        assert_eq!(registry.get_all_tool_specs().len(), 2);
    }

    // register_dynamic_tool: a dynamic tool overrides a base tool of the same name
    #[test]
    fn dynamic_tool_overrides_base_by_name() {
        let mut registry = ToolRegistry::new();
        registry
            .add(MockTool::arc_with_description("shared", "base description"))
            .unwrap();
        registry
            .register_dynamic_tool(MockTool::arc_with_description(
                "shared",
                "dynamic description",
            ))
            .unwrap();

        // Exactly one "shared" spec, and it is the dynamic one.
        let specs: Vec<_> = registry
            .tool_specs()
            .into_iter()
            .filter(|spec| spec.name == "shared")
            .collect();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].description, "dynamic description");
        assert_eq!(
            registry.get("shared").unwrap().description(),
            "dynamic description"
        );

        // Re-registering a dynamic tool of the same name replaces the prior one.
        registry
            .register_dynamic_tool(MockTool::arc_with_description("shared", "newer"))
            .unwrap();
        assert_eq!(registry.dynamic_tools().len(), 1);
        assert_eq!(registry.get("shared").unwrap().description(), "newer");
    }

    // fork: shares the base, but dynamic tools added to the fork do not leak back
    #[test]
    fn fork_shares_base_and_isolates_dynamic() {
        let mut original = ToolRegistry::new();
        original.add(MockTool::arc("base-tool")).unwrap();

        let mut forked = original.fork();
        forked
            .register_dynamic_tool(MockTool::arc("fork-only"))
            .unwrap();

        // Base is visible through the fork (shared, read-only).
        assert!(forked.get("base-tool").is_some());
        // The dynamic tool is on the fork only.
        assert!(forked.get("fork-only").is_some());
        assert!(original.get("fork-only").is_none());
        assert_eq!(original.dynamic_tools().len(), 0);
    }
}

//! Validator agent for input/output validation and data quality checks.

use async_trait::async_trait;
use crate::models::graph::Agent;
use crate::models::tools::ToolRegistryTrait;
use regex::Regex;

/// Validation rule for checking inputs/outputs
#[derive(Clone)]
pub struct ValidationRule {
    /// Name of the rule
    pub name: String,
    /// Validation function
    pub validator: Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>,
    /// Whether this rule is critical (stops execution if failed)
    pub critical: bool,
}

use std::sync::Arc;

/// Validator agent that validates inputs before processing.
/// 
/// # Example
/// ```rust
/// use llmgraph::agents::ValidatorAgent;
/// use llmgraph::agents::validator::ValidationRule;
/// use std::sync::Arc;
/// 
/// let validator = ValidatorAgent::new()
///     .add_rule(ValidationRule {
///         name: "length_check".to_string(),
///         validator: Arc::new(|input| {
///             if input.len() > 1000 {
///                 Err("Input too long".to_string())
///             } else {
///                 Ok(())
///             }
///         }),
///         critical: true,
///     })
///     .add_rule(ValidationRule {
///         name: "no_profanity".to_string(),
///         validator: Arc::new(|input| {
///             if input.contains("bad_word") {
///                 Err("Profanity detected".to_string())
///             } else {
///                 Ok(())
///             }
///         }),
///         critical: false,
///     });
/// ```
pub struct ValidatorAgent {
    rules: Vec<ValidationRule>,
    name: String,
    next_node_on_success: Option<i32>,
    next_node_on_failure: Option<i32>,
    strict_mode: bool,
}

impl ValidatorAgent {
    /// Create a new validator agent
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            name: "Validator".to_string(),
            next_node_on_success: None,
            next_node_on_failure: None,
            strict_mode: false,
        }
    }

    /// Add a validation rule
    pub fn add_rule(mut self, rule: ValidationRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add a regex pattern validation rule
    pub fn add_pattern_rule(
        mut self,
        name: impl Into<String>,
        pattern: impl Into<String>,
        error_msg: impl Into<String>,
        critical: bool,
    ) -> Self {
        let pattern_str = pattern.into();
        let error = error_msg.into();
        let rule = ValidationRule {
            name: name.into(),
            validator: Arc::new(move |input| {
                if let Ok(re) = Regex::new(&pattern_str) {
                    if !re.is_match(input) {
                        return Err(error.clone());
                    }
                }
                Ok(())
            }),
            critical,
        };
        self.rules.push(rule);
        self
    }

    /// Add a length validation rule
    pub fn add_length_rule(mut self, min: Option<usize>, max: Option<usize>, critical: bool) -> Self {
        let rule = ValidationRule {
            name: "length_check".to_string(),
            validator: Arc::new(move |input| {
                if let Some(min_len) = min {
                    if input.len() < min_len {
                        return Err(format!("Input too short (min: {})", min_len));
                    }
                }
                if let Some(max_len) = max {
                    if input.len() > max_len {
                        return Err(format!("Input too long (max: {})", max_len));
                    }
                }
                Ok(())
            }),
            critical,
        };
        self.rules.push(rule);
        self
    }

    /// Set the next node on validation success
    pub fn with_success_route(mut self, node_id: i32) -> Self {
        self.next_node_on_success = Some(node_id);
        self
    }

    /// Set the next node on validation failure
    pub fn with_failure_route(mut self, node_id: i32) -> Self {
        self.next_node_on_failure = Some(node_id);
        self
    }

    /// Enable strict mode (all rules must pass)
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Set the agent name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Validate input against all rules
    fn validate(&self, input: &str) -> Result<Vec<String>, Vec<String>> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        for rule in &self.rules {
            match (rule.validator)(input) {
                Ok(()) => {}
                Err(msg) => {
                    let error_msg = format!("{}: {}", rule.name, msg);
                    if rule.critical || self.strict_mode {
                        errors.push(error_msg);
                    } else {
                        warnings.push(error_msg);
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(warnings)
        } else {
            Err(errors)
        }
    }
}

impl Default for ValidatorAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for ValidatorAgent {
    async fn run(
        &mut self,
        input: &str,
        _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
        match self.validate(input) {
            Ok(warnings) => {
                let mut response = format!("Validation passed for: {}", input);
                if !warnings.is_empty() {
                    response.push_str(&format!("\nWarnings: {}", warnings.join(", ")));
                }
                (response, self.next_node_on_success)
            }
            Err(errors) => {
                let response = format!(
                    "Validation failed for: {}\nErrors: {}",
                    input,
                    errors.join(", ")
                );
                (response, self.next_node_on_failure)
            }
        }
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}
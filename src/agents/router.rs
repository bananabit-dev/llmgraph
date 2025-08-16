//! Router agent for conditional routing based on input patterns.

use async_trait::async_trait;
use crate::models::graph::Agent;
use crate::models::tools::ToolRegistryTrait;
use regex::Regex;

/// Configuration for a routing rule
#[derive(Clone)]
pub struct RouteRule {
    /// Pattern to match (regex)
    pub pattern: String,
    /// Node ID to route to if pattern matches
    pub target_node: i32,
    /// Optional description of this route
    pub description: Option<String>,
}

/// Router agent that routes messages based on pattern matching.
/// 
/// # Example
/// ```rust
/// use llmgraph::agents::RouterAgent;
/// use llmgraph::agents::router::RouteRule;
/// 
/// let mut router = RouterAgent::new()
///     .add_route(RouteRule {
///         pattern: r"technical|bug|error".to_string(),
///         target_node: 1,
///         description: Some("Route to technical support".to_string()),
///     })
///     .add_route(RouteRule {
///         pattern: r"sales|pricing|buy".to_string(),
///         target_node: 2,
///         description: Some("Route to sales team".to_string()),
///     })
///     .set_default(3); // Default route
/// ```
pub struct RouterAgent {
    routes: Vec<RouteRule>,
    default_route: Option<i32>,
    case_sensitive: bool,
    name: String,
}

impl RouterAgent {
    /// Create a new router agent
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            default_route: None,
            case_sensitive: false,
            name: "Router".to_string(),
        }
    }

    /// Add a routing rule
    pub fn add_route(mut self, rule: RouteRule) -> Self {
        self.routes.push(rule);
        self
    }

    /// Set the default route for unmatched inputs
    pub fn set_default(mut self, node_id: i32) -> Self {
        self.default_route = Some(node_id);
        self
    }

    /// Set whether pattern matching is case-sensitive
    pub fn set_case_sensitive(mut self, sensitive: bool) -> Self {
        self.case_sensitive = sensitive;
        self
    }

    /// Set the agent name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Find the best matching route for the input
    fn find_route(&self, input: &str) -> Option<i32> {
        let test_input = if self.case_sensitive {
            input.to_string()
        } else {
            input.to_lowercase()
        };

        for rule in &self.routes {
            let pattern = if self.case_sensitive {
                &rule.pattern
            } else {
                &rule.pattern.to_lowercase()
            };

            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(&test_input) {
                    return Some(rule.target_node);
                }
            }
        }

        self.default_route
    }
}

impl Default for RouterAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for RouterAgent {
    async fn run(
        &mut self,
        input: &str,
        _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
        let route = self.find_route(input);
        
        let response = match route {
            Some(node_id) => {
                format!("Routing to node {}: {}", node_id, input)
            }
            None => {
                format!("No matching route found for: {}", input)
            }
        };

        (response, route)
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}
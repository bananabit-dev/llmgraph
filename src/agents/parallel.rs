//! Parallel agent for executing multiple agents concurrently.

use async_trait::async_trait;
use crate::models::graph::Agent;
use crate::models::tools::ToolRegistryTrait;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Strategy for combining results from parallel agents
#[derive(Clone)]
pub enum CombineStrategy {
    /// Concatenate all outputs
    Concatenate,
    /// Return first non-empty result
    FirstValid,
    /// Return all results as JSON array
    JsonArray,
    /// Custom combiner function
    Custom(Arc<dyn Fn(Vec<String>) -> String + Send + Sync>),
}

/// Parallel agent that runs multiple agents concurrently.
///
/// # Example
/// ```rust,ignore
/// use llmgraph::agents::ParallelAgent;
/// use llmgraph::agents::parallel::CombineStrategy;
///
/// // Assuming agent1, agent2, agent3 are defined
/// let parallel = ParallelAgent::new()
///     .add_agent(agent1)
///     .add_agent(agent2)
///     .add_agent(agent3)
///     .with_strategy(CombineStrategy::JsonArray);
/// ```
pub struct ParallelAgent {
    agents: Vec<Arc<Mutex<Box<dyn Agent>>>>,
    strategy: CombineStrategy,
    name: String,
    timeout: Option<tokio::time::Duration>,
}

impl ParallelAgent {
    /// Create a new parallel agent
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            strategy: CombineStrategy::Concatenate,
            name: "Parallel".to_string(),
            timeout: None,
        }
    }

    /// Add an agent to run in parallel
    pub fn add_agent(mut self, agent: Box<dyn Agent>) -> Self {
        self.agents.push(Arc::new(Mutex::new(agent)));
        self
    }

    /// Set the combine strategy
    pub fn with_strategy(mut self, strategy: CombineStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set execution timeout
    pub fn with_timeout(mut self, timeout: tokio::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the agent name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Combine results based on strategy
    fn combine_results(&self, results: Vec<String>) -> String {
        match &self.strategy {
            CombineStrategy::Concatenate => results.join("\n"),
            CombineStrategy::FirstValid => {
                results.into_iter()
                    .find(|s| !s.is_empty())
                    .unwrap_or_default()
            }
            CombineStrategy::JsonArray => {
                serde_json::json!(results).to_string()
            }
            CombineStrategy::Custom(combiner) => combiner(results),
        }
    }
}

impl Default for ParallelAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for ParallelAgent {
    async fn run(
        &mut self,
        input: &str,
        tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
        if self.agents.is_empty() {
            return ("No agents configured for parallel execution".to_string(), None);
        }

        // Create futures for all agents
        let mut futures = Vec::new();
        for agent_mutex in &self.agents {
            let input_clone = input.to_string();
            let agent = agent_mutex.clone();
            
            let future = async move {
                let mut agent_guard = agent.lock().await;
                agent_guard.run(&input_clone, tool_registry).await.0
            };
            
            futures.push(future);
        }

        // Execute all agents in parallel
        let results = if let Some(timeout) = self.timeout {
            match tokio::time::timeout(timeout, futures::future::join_all(futures)).await {
                Ok(results) => results,
                Err(_) => {
                    return ("Parallel execution timed out".to_string(), None);
                }
            }
        } else {
            futures::future::join_all(futures).await
        };

        let combined = self.combine_results(results);
        (combined, None)
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

// Re-export futures for convenience
use futures;
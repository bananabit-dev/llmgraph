//! Stateful agent with memory and context persistence.

use async_trait::async_trait;
use crate::models::graph::Agent;
use crate::models::tools::ToolRegistryTrait;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// State that can be persisted between agent executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Key-value store for arbitrary data
    pub data: HashMap<String, serde_json::Value>,
    /// Conversation history
    pub history: Vec<String>,
    /// Current context
    pub context: Option<String>,
    /// Execution count
    pub execution_count: usize,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
            history: Vec::new(),
            context: None,
            execution_count: 0,
        }
    }
}

/// A stateful agent that maintains memory between executions.
/// 
/// # Example
/// ```rust
/// use llmgraph::agents::StatefulAgent;
/// 
/// let agent = StatefulAgent::new("MemoryAgent")
///     .with_processor(Box::new(|input, state| {
///         // Access and modify state
///         state.execution_count += 1;
///         state.history.push(input.to_string());
///         
///         let response = format!(
///             "Processed {} (execution #{})",
///             input, state.execution_count
///         );
///         
///         (response, None) // Return response and optional next node
///     }));
/// ```
pub struct StatefulAgent {
    name: String,
    state: Arc<Mutex<AgentState>>,
    processor: Option<Box<dyn Fn(&str, &mut AgentState) -> (String, Option<i32>) + Send + Sync>>,
    max_history: usize,
}

impl StatefulAgent {
    /// Create a new stateful agent
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: Arc::new(Mutex::new(AgentState::default())),
            processor: None,
            max_history: 100,
        }
    }

    /// Set the processing function
    pub fn with_processor(
        mut self,
        processor: Box<dyn Fn(&str, &mut AgentState) -> (String, Option<i32>) + Send + Sync>,
    ) -> Self {
        self.processor = Some(processor);
        self
    }

    /// Set maximum history size
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Get the current state
    pub fn get_state(&self) -> AgentState {
        self.state.lock().unwrap().clone()
    }

    /// Set the state
    pub fn set_state(&mut self, state: AgentState) {
        *self.state.lock().unwrap() = state;
    }

    /// Store a value in the state
    pub fn store(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.state.lock().unwrap().data.insert(key.into(), value);
    }

    /// Retrieve a value from the state
    pub fn retrieve(&self, key: &str) -> Option<serde_json::Value> {
        self.state.lock().unwrap().data.get(key).cloned()
    }

    /// Clear the state
    pub fn clear_state(&mut self) {
        *self.state.lock().unwrap() = AgentState::default();
    }

    /// Save state to JSON
    pub fn save_state(&self) -> Result<String, serde_json::Error> {
        let state = self.state.lock().unwrap();
        serde_json::to_string(&*state)
    }

    /// Load state from JSON
    pub fn load_state(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let state: AgentState = serde_json::from_str(json)?;
        *self.state.lock().unwrap() = state;
        Ok(())
    }
}

#[async_trait]
impl Agent for StatefulAgent {
    async fn run(
        &mut self,
        input: &str,
        _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
        let mut state = self.state.lock().unwrap();
        
        // Update state
        state.execution_count += 1;
        state.history.push(input.to_string());
        
        // Trim history if it exceeds max size
        if state.history.len() > self.max_history {
            state.history.remove(0);
        }

        // Process with custom processor if provided
        if let Some(processor) = &self.processor {
            let result = processor(input, &mut state);
            drop(state); // Release lock
            return result;
        }

        // Default processing
        let response = format!(
            "Stateful agent processed: {} (execution #{}, history: {} items)",
            input,
            state.execution_count,
            state.history.len()
        );
        
        drop(state); // Release lock
        (response, None)
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}
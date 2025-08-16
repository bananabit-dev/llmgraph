//! Graph-based multi-agent system for AI workflows
//!
//! This module provides the core graph structure and agent trait for building
//! conversational AI applications with function calling capabilities.

use async_trait::async_trait;
use std::collections::HashMap;
use crate::models::tools::{ToolRegistry, Tool, ToolRegistryTrait, CombinedToolRegistry};

/// Trait for implementing agents that can process inputs and communicate within the graph.
///
/// # Example
/// ```rust
/// use async_trait::async_trait;
/// use llmgraph::models::graph::Agent;
/// use llmgraph::models::tools::ToolRegistryTrait;
///
/// pub struct MyAgent;
///
/// #[async_trait]
/// impl Agent for MyAgent {
///     async fn run(
///         &mut self,
///         input: &str,
///         tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
///     ) -> (String, Option<i32>) {
///         let response = format!("Processed: {}", input);
///         (response, None) // None ends the chain
///     }
///
///     fn get_name(&self) -> &str {
///         "MyAgent"
///     }
/// }
/// ```
#[async_trait]
pub trait Agent: Send {
    /// Process input and optionally route to the next agent.
    ///
    /// # Arguments
    /// * `input` - The input string to process
    /// * `tool_registry` - Registry containing available tools/functions
    ///
    /// # Returns
    /// A tuple containing:
    /// * `String` - The processed output
    /// * `Option<i32>` - The ID of the next agent to route to (None terminates)
    async fn run(
        &mut self,
        input: &str,
        tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>);
    
    /// Get the name of this agent for identification purposes.
    fn get_name(&self) -> &str;
}

/// A directed graph of agents that can communicate and pass messages.
///
/// The graph supports:
/// - Dynamic agent registration
/// - Tool registration (both global and node-specific)
/// - Sequential message passing through connected agents
///
/// # Example
/// ```rust,no_run
/// use llmgraph::Graph;
/// use llmgraph::Agent;
/// use async_trait::async_trait;
///
/// struct SimpleAgent;
///
/// #[async_trait]
/// impl Agent for SimpleAgent {
///     async fn run(&mut self, input: &str, _: &(dyn llmgraph::ToolRegistryTrait + Send + Sync)) -> (String, Option<i32>) {
///         (format!("Processed: {}", input), None)
///     }
///     fn get_name(&self) -> &str { "Simple" }
/// }
///
/// #[tokio::main]
/// async fn main() {
///     let mut graph = Graph::new();
///     graph.add_node(0, Box::new(SimpleAgent));
///     graph.add_node(1, Box::new(SimpleAgent));
///     graph.add_edge(0, 1).unwrap();
///
///     let result = graph.run(0, "Hello").await;
///     println!("Result: {}", result);
/// }
/// ```
pub struct Graph {
    nodes: HashMap<i32, Node>,
    tool_registry: ToolRegistry, // Shared tool registry
}

struct Node {
    agent: Box<dyn Agent>,
    neighbors: Vec<i32>,
    tool_registry: ToolRegistry, // Node-specific tool registry
}

impl Graph {
    /// Create a new empty graph.
    ///
    /// # Example
    /// ```rust
    /// use llmgraph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// ```
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            tool_registry: ToolRegistry::new(),
        }
    }

    /// Register a tool globally (available to all agents).
    ///
    /// # Arguments
    /// * `tool` - The tool definition
    /// * `function` - The function implementation
    ///
    /// # Example
    /// ```rust,ignore
    /// # use llmgraph::{Graph, Tool};
    /// # let mut graph = Graph::new();
    /// # let weather_tool = Tool { /* ... */ };
    /// graph.register_tool(weather_tool, |args| {
    ///     Ok(serde_json::json!({"temperature": 72}))
    /// });
    /// ```
    pub fn register_tool<F>(&mut self, tool: Tool, function: F)
    where
        F: Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync + 'static,
    {
        self.tool_registry.register_tool(tool, function);
    }

    /// Register a tool for a specific node only.
    ///
    /// # Arguments
    /// * `node_id` - The ID of the node
    /// * `tool` - The tool definition
    /// * `function` - The function implementation
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err(String)` if the node doesn't exist
    ///
    /// # Example
    /// ```rust,ignore
    /// # use llmgraph::{Graph, Tool};
    /// # let mut graph = Graph::new();
    /// # let calculator_tool = Tool { /* ... */ };
    /// graph.register_tool_for_node(0, calculator_tool, |args| {
    ///     Ok(serde_json::json!({"result": 42}))
    /// }).unwrap();
    /// ```
    pub fn register_tool_for_node<F>(
        &mut self,
        node_id: i32,
        tool: Tool,
        function: F,
    ) -> Result<(), String>
    where
        F: Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync + 'static,
    {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.tool_registry.register_tool(tool, function);
            Ok(())
        } else {
            Err(format!("Node {} does not exist", node_id))
        }
    }

    /// Get the tool registry for a specific node.
    ///
    /// # Arguments
    /// * `node_id` - The ID of the node
    ///
    /// # Returns
    /// * `Some(&ToolRegistry)` if the node exists
    /// * `None` if the node doesn't exist
    pub fn get_node_tool_registry(&self, node_id: i32) -> Option<&ToolRegistry> {
        self.nodes.get(&node_id).map(|node| &node.tool_registry)
    }

    /// Get the shared global tool registry.
    ///
    /// # Returns
    /// A reference to the global tool registry
    pub fn get_shared_tool_registry(&self) -> &ToolRegistry {
        &self.tool_registry
    }

    /// Add a new agent node to the graph.
    ///
    /// # Arguments
    /// * `id` - The unique identifier for the node
    /// * `agent` - The agent to add
    ///
    /// # Example
    /// ```rust,ignore
    /// # use llmgraph::Graph;
    /// # let mut graph = Graph::new();
    /// # let my_agent = MyAgent;
    /// graph.add_node(0, Box::new(my_agent));
    /// ```
    pub fn add_node(&mut self, id: i32, agent: Box<dyn Agent>) {
        self.nodes.insert(
            id,
            Node {
                agent,
                neighbors: Vec::new(),
                tool_registry: ToolRegistry::new(),
            },
        );
    }

    /// Add a bidirectional edge between two nodes.
    ///
    /// # Arguments
    /// * `u` - The first node ID
    /// * `v` - The second node ID
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err(String)` if one or both nodes don't exist
    ///
    /// # Example
    /// ```rust,ignore
    /// # use llmgraph::Graph;
    /// # let mut graph = Graph::new();
    /// # // Assuming nodes 0 and 1 exist
    /// graph.add_edge(0, 1).unwrap();
    /// ```
    pub fn add_edge(&mut self, u: i32, v: i32) -> Result<(), String> {
        if !self.nodes.contains_key(&u) || !self.nodes.contains_key(&v) {
            return Err("One or both nodes do not exist".to_string());
        }

        if let Some(node) = self.nodes.get_mut(&u) {
            if !node.neighbors.contains(&v) {
                node.neighbors.push(v);
            }
        }

        if let Some(node) = self.nodes.get_mut(&v) {
            if !node.neighbors.contains(&u) {
                node.neighbors.push(u);
            }
        }

        Ok(())
    }

    /// Print the graph structure to stdout for debugging.
    ///
    /// # Example
    /// ```rust,ignore
    /// # use llmgraph::Graph;
    /// # let graph = Graph::new();
    /// graph.print();
    /// // Output:
    /// // Adjacency list for the Graph:
    /// // 0 (Agent: Manager) -> 1 2
    /// // 1 (Agent: Developer) -> 0
    /// ```
    pub fn print(&self) {
        println!("Adjacency list for the Graph:");
        for (id, node) in &self.nodes {
            print!("{} (Agent: {}) -> ", id, node.agent.get_name());
            for neighbor in &node.neighbors {
                print!("{} ", neighbor);
            }
            println!();
        }
    }

    /// Execute the graph starting from a specific node.
    ///
    /// This method will:
    /// 1. Start execution at the specified node
    /// 2. Pass the output of each agent to the next one
    /// 3. Continue until an agent returns None for the next node
    ///
    /// # Arguments
    /// * `start_id` - The ID of the starting node
    /// * `input` - The initial input string
    ///
    /// # Returns
    /// A string containing the accumulated output from all agents
    ///
    /// # Example
    /// ```rust,ignore
    /// # use llmgraph::Graph;
    /// # let mut graph = Graph::new();
    /// # // Async context required
    /// let result = graph.run(0, "Process this task").await;
    /// println!("Result: {}", result);
    /// ```
    pub async fn run(&mut self, start_id: i32, input: &str) -> String {
        let mut current_id = start_id;
        let mut current_input = input.to_string();
        let mut result = String::new();

        loop {
            // First, check if the current node exists
            if !self.nodes.contains_key(&current_id) {
                result.push_str(&format!("Error: Node {} does not exist\n", current_id));
                break;
            }

            // Use unsafe to work around the borrowing issue
            // This is safe because we're not modifying the structure of the HashMap,
            // only calling a method on one of its values
            let (output, next_id) = unsafe {
                // Get raw pointers to avoid borrowing conflicts
                let nodes_ptr = &mut self.nodes as *mut HashMap<i32, Node>;
                let tool_registry_ptr = &self.tool_registry as *const ToolRegistry;
                
                // Get the node's tool registry reference
                let node_tool_registry_ptr = {
                    let nodes_ref = &*nodes_ptr;
                    &nodes_ref[&current_id].tool_registry as *const ToolRegistry
                };
                
                // Create combined registry using the raw pointers
                let combined_registry = CombinedToolRegistry::new(
                    &*node_tool_registry_ptr as &(dyn ToolRegistryTrait + Send + Sync),
                    &*tool_registry_ptr as &(dyn ToolRegistryTrait + Send + Sync),
                );
                
                // Now get mutable access to the node and run the agent
                let nodes_mut = &mut *nodes_ptr;
                let node = nodes_mut.get_mut(&current_id).unwrap();
                node.agent.run(&current_input, &combined_registry).await
            };

            result.push_str(&output);
            result.push('\n');

            match next_id {
                Some(next) => {
                    if self.nodes.contains_key(&next) {
                        current_id = next;
                        current_input = output;
                    } else {
                        result.push_str(&format!("Error: Node {} does not exist\n", next));
                        break;
                    }
                }
                None => break,
            }
        }

        result
    }
}
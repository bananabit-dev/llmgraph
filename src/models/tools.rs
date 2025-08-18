//! Tool and message definitions for function calling in AI agents.
//!
//! This module provides the data structures and traits needed for
//! implementing function calling capabilities in AI agents.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// -----------------------------
// Tool definitions
// -----------------------------

/// Represents a tool/function that can be called by an AI agent.
///
/// # Example
/// ```rust
/// use llmgraph::models::tools::{Tool, Function, Parameters, Property};
/// use std::collections::HashMap;
///
/// let tool = Tool {
///     tool_type: "function".to_string(),
///     function: Function {
///         name: "get_weather".to_string(),
///         description: "Get weather information".to_string(),
///         parameters: Parameters {
///             param_type: "object".to_string(),
///             properties: HashMap::new(),
///             required: vec![],
///         },
///     },
/// };
/// ```
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    /// The type of tool (typically "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition
    pub function: Function,
}

/// Defines a function that can be called by an AI agent.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Function {
    /// The name of the function
    pub name: String,
    /// A description of what the function does
    pub description: String,
    /// The parameters that the function accepts
    pub parameters: Parameters,
}

/// Defines the parameters for a function.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Parameters {
    /// The type of parameters (typically "object")
    #[serde(rename = "type")]
    pub param_type: String,
    /// Map of parameter names to their properties
    pub properties: HashMap<String, Property>,
    /// List of required parameter names
    pub required: Vec<String>,
}

/// Defines a single parameter property.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Property {
    /// The type of the property (e.g., "string", "number", "array")
    #[serde(rename = "type")]
    pub prop_type: String,
    /// Optional description of the property
    pub description: Option<String>,
    /// For array types, defines the items in the array
    pub items: Option<Box<Property>>,
}

/// Represents a tool call request from an AI model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: String,
    /// The type of call (typically "function")
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function call details
    pub function: FunctionCall,
}

/// Details of a function call.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    /// The name of the function to call
    pub name: String,
    /// The arguments as a JSON string
    pub arguments: String,
}

/// Represents a message in a conversation.
///
/// # Example
/// ```rust
/// use llmgraph::models::tools::Message;
///
/// let message = Message {
///     role: "user".to_string(),
///     content: Some("Hello, AI!".to_string()),
///     tool_calls: None,
/// };
/// ```
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    /// The role of the message sender ("system", "user", "assistant", "tool")
    pub role: String,
    /// The text content of the message
    pub content: Option<String>,
    /// Optional tool calls made by the assistant
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolResultMessage {
    pub role: String,
    #[serde(rename = "tool_call_id")]
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LLMResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
}

// -----------------------------
// Tool Registry Trait
// -----------------------------
/// Trait for implementing tool registries.
///
/// A tool registry manages available tools and their execution.
pub trait ToolRegistryTrait: Send + Sync {
    /// Get all available tools in this registry.
    ///
    /// # Returns
    /// A vector of all registered tools
    fn get_tools(&self) -> Vec<Tool>;
    
    /// Execute a tool by name with the given arguments.
    ///
    /// # Arguments
    /// * `name` - The name of the tool to execute
    /// * `arguments` - JSON string containing the arguments
    ///
    /// # Returns
    /// * `Ok(Value)` - The result of the tool execution
    /// * `Err(String)` - An error message if execution fails
    fn execute_tool(&self, name: &str, arguments: &str) -> Result<Value, String>;
}

// -----------------------------
// Tool Registry Implementation
// -----------------------------
/// Type alias for tool functions.
pub type ToolFunction = dyn Fn(Value) -> Result<Value, String> + Send + Sync;

/// A registry for managing tools and their implementations.
///
/// # Example
/// ```rust,ignore
/// use llmgraph::models::tools::ToolRegistry;
///
/// let mut registry = ToolRegistry::new();
/// // Assuming my_tool is defined
/// registry.register_tool(my_tool, |args| {
///     Ok(serde_json::json!({"result": "success"}))
/// });
/// ```
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
    functions: HashMap<String, Box<ToolFunction>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    /// Register a tool with its implementation.
    ///
    /// # Arguments
    /// * `tool` - The tool definition
    /// * `function` - The function that implements the tool
    ///
    /// # Example
    /// ```rust,ignore
    /// use serde_json::json;
    /// // Assuming registry and weather_tool are defined
    /// registry.register_tool(weather_tool, |args| {
    ///     let location = args["location"].as_str().unwrap();
    ///     Ok(json!({"temp": 72, "location": location}))
    /// });
    /// ```
    pub fn register_tool<F>(&mut self, tool: Tool, function: F)
    where
        F: Fn(Value) -> Result<Value, String> + Send + Sync + 'static,
    {
        let name = tool.function.name.clone();
        self.tools.insert(name.clone(), tool);
        self.functions.insert(name, Box::new(function));
    }
}

impl ToolRegistryTrait for ToolRegistry {
    fn get_tools(&self) -> Vec<Tool> {
        self.tools.values().cloned().collect()
    }

    fn execute_tool(&self, name: &str, arguments: &str) -> Result<Value, String> {
        let args: Value = serde_json::from_str(arguments)
            .map_err(|e| format!("Failed to parse arguments: {}", e))?;
        if let Some(function) = self.functions.get(name) {
            function(args)
        } else {
            Err(format!("Tool '{}' not found", name))
        }
    }
}

// -----------------------------
// Combined Tool Registry
// -----------------------------
/// A registry that combines two tool registries.
///
/// The primary registry takes precedence over the secondary registry
/// when tools have the same name.
pub struct CombinedToolRegistry<'a> {
    primary: &'a dyn ToolRegistryTrait,
    secondary: &'a dyn ToolRegistryTrait,
}

impl<'a> CombinedToolRegistry<'a> {
    /// Create a new combined registry.
    ///
    /// # Arguments
    /// * `primary` - The primary registry (higher priority)
    /// * `secondary` - The secondary registry (lower priority)
    pub fn new(primary: &'a dyn ToolRegistryTrait, secondary: &'a dyn ToolRegistryTrait) -> Self {
        Self { primary, secondary }
    }
}

impl<'a> ToolRegistryTrait for CombinedToolRegistry<'a> {
    fn get_tools(&self) -> Vec<Tool> {
        let mut tools = self.secondary.get_tools();
        let primary_tools = self.primary.get_tools();

        // Remove duplicates from secondary
        tools.retain(|tool| {
            !primary_tools.iter().any(|t| t.function.name == tool.function.name)
        });

        // Add primary tools (take precedence)
        tools.extend(primary_tools);
        tools
    }

    fn execute_tool(&self, name: &str, arguments: &str) -> Result<Value, String> {
        if let Ok(result) = self.primary.execute_tool(name, arguments) {
            return Ok(result);
        }
        self.secondary.execute_tool(name, arguments)
    }
}
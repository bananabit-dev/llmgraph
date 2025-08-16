//! Error handling module for the LLMGraph library.
//! 
//! Provides custom error types for better error handling and debugging.

use std::fmt;
use std::error::Error;

/// Main error type for the LLMGraph library
#[derive(Debug)]
pub enum LLMGraphError {
    /// Node-related errors
    NodeError(NodeError),
    /// Tool execution errors
    ToolError(ToolError),
    /// API/Network errors
    ApiError(ApiError),
    /// Agent execution errors
    AgentError(AgentError),
    /// Graph structure errors
    GraphError(GraphError),
    /// Serialization/Deserialization errors
    SerializationError(String),
}

/// Errors related to node operations
#[derive(Debug)]
pub enum NodeError {
    NodeNotFound(i32),
    NodeAlreadyExists(i32),
    InvalidNodeId(i32),
}

/// Errors related to tool operations
#[derive(Debug)]
pub enum ToolError {
    ToolNotFound(String),
    ToolExecutionFailed { name: String, error: String },
    InvalidArguments { name: String, error: String },
    ToolAlreadyRegistered(String),
}

/// Errors related to API calls
#[derive(Debug)]
pub enum ApiError {
    RequestFailed(String),
    InvalidResponse(String),
    RateLimitExceeded,
    AuthenticationFailed,
    Timeout,
}

/// Errors related to agent execution
#[derive(Debug)]
pub enum AgentError {
    ExecutionFailed { agent: String, error: String },
    InvalidInput(String),
    InvalidOutput(String),
    MaxIterationsExceeded,
}

/// Errors related to graph structure
#[derive(Debug)]
pub enum GraphError {
    CycleDetected,
    DisconnectedGraph,
    InvalidEdge { from: i32, to: i32 },
    EmptyGraph,
}

impl fmt::Display for LLMGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMGraphError::NodeError(e) => write!(f, "Node error: {}", e),
            LLMGraphError::ToolError(e) => write!(f, "Tool error: {}", e),
            LLMGraphError::ApiError(e) => write!(f, "API error: {}", e),
            LLMGraphError::AgentError(e) => write!(f, "Agent error: {}", e),
            LLMGraphError::GraphError(e) => write!(f, "Graph error: {}", e),
            LLMGraphError::SerializationError(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeError::NodeNotFound(id) => write!(f, "Node {} not found", id),
            NodeError::NodeAlreadyExists(id) => write!(f, "Node {} already exists", id),
            NodeError::InvalidNodeId(id) => write!(f, "Invalid node ID: {}", id),
        }
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::ToolNotFound(name) => write!(f, "Tool '{}' not found", name),
            ToolError::ToolExecutionFailed { name, error } => {
                write!(f, "Tool '{}' execution failed: {}", name, error)
            }
            ToolError::InvalidArguments { name, error } => {
                write!(f, "Invalid arguments for tool '{}': {}", name, error)
            }
            ToolError::ToolAlreadyRegistered(name) => {
                write!(f, "Tool '{}' is already registered", name)
            }
        }
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::RequestFailed(msg) => write!(f, "API request failed: {}", msg),
            ApiError::InvalidResponse(msg) => write!(f, "Invalid API response: {}", msg),
            ApiError::RateLimitExceeded => write!(f, "API rate limit exceeded"),
            ApiError::AuthenticationFailed => write!(f, "API authentication failed"),
            ApiError::Timeout => write!(f, "API request timed out"),
        }
    }
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::ExecutionFailed { agent, error } => {
                write!(f, "Agent '{}' execution failed: {}", agent, error)
            }
            AgentError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            AgentError::InvalidOutput(msg) => write!(f, "Invalid output: {}", msg),
            AgentError::MaxIterationsExceeded => write!(f, "Maximum iterations exceeded"),
        }
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::CycleDetected => write!(f, "Cycle detected in graph"),
            GraphError::DisconnectedGraph => write!(f, "Graph is disconnected"),
            GraphError::InvalidEdge { from, to } => {
                write!(f, "Invalid edge from {} to {}", from, to)
            }
            GraphError::EmptyGraph => write!(f, "Graph is empty"),
        }
    }
}

impl Error for LLMGraphError {}
impl Error for NodeError {}
impl Error for ToolError {}
impl Error for ApiError {}
impl Error for AgentError {}
impl Error for GraphError {}

/// Result type alias for LLMGraph operations
pub type LLMGraphResult<T> = Result<T, LLMGraphError>;

// Conversion implementations
impl From<NodeError> for LLMGraphError {
    fn from(error: NodeError) -> Self {
        LLMGraphError::NodeError(error)
    }
}

impl From<ToolError> for LLMGraphError {
    fn from(error: ToolError) -> Self {
        LLMGraphError::ToolError(error)
    }
}

impl From<ApiError> for LLMGraphError {
    fn from(error: ApiError) -> Self {
        LLMGraphError::ApiError(error)
    }
}

impl From<AgentError> for LLMGraphError {
    fn from(error: AgentError) -> Self {
        LLMGraphError::AgentError(error)
    }
}

impl From<GraphError> for LLMGraphError {
    fn from(error: GraphError) -> Self {
        LLMGraphError::GraphError(error)
    }
}

impl From<serde_json::Error> for LLMGraphError {
    fn from(error: serde_json::Error) -> Self {
        LLMGraphError::SerializationError(error.to_string())
    }
}
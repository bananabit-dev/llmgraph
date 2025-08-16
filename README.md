# LLMGraph - Multi-Agent AI System Framework

A Rust library for building conversational AI applications with function calling capabilities. LLMGraph allows you to create directed graphs of AI agents that can communicate with each other, use tools, and process tasks in a coordinated manner.

## Features

- **Agent-based Architecture**: Create custom agents that can process inputs and communicate with other agents
- **Tool/Function Calling**: Register tools that agents can use during execution
- **Graph-based Workflow**: Build complex workflows by connecting agents in a directed graph
- **Flexible Tool Registry**: Support for both global and node-specific tools
- **Async/Await Support**: Fully asynchronous execution for optimal performance
- **OpenRouter Integration**: Built-in support for LLM APIs through OpenRouter

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
llmgraph = "0.1.0"
```

## Quick Start

### Creating a Simple Agent

```rust
use llmgraph::models::graph::{Agent, Graph};
use llmgraph::models::tools::ToolRegistryTrait;
use async_trait::async_trait;

pub struct GreeterAgent;

#[async_trait]
impl Agent for GreeterAgent {
    async fn run(
        &mut self,
        input: &str,
        _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
        let response = format!("Hello! You said: {}", input);
        (response, None) // None means end of chain
    }

    fn get_name(&self) -> &str {
        "Greeter"
    }
}
```

### Building an Agent Graph

```rust
#[tokio::main]
async fn main() {
    // Create a new graph
    let mut graph = Graph::new();
    
    // Add agents to the graph
    graph.add_node(0, Box::new(ManagerAgent));
    graph.add_node(1, Box::new(DeveloperAgent));
    graph.add_node(2, Box::new(ReviewerAgent));
    
    // Connect the agents
    graph.add_edge(0, 1).unwrap(); // Manager -> Developer
    graph.add_edge(1, 2).unwrap(); // Developer -> Reviewer
    
    // Run the graph
    let result = graph.run(0, "Create a new feature").await;
    println!("Result: {}", result);
}
```

### Working with Tools

```rust
use llmgraph::models::tools::{Tool, Function, Parameters, Property};
use std::collections::HashMap;

// Define a weather tool
fn create_weather_tool() -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            parameters: Parameters {
                param_type: "object".to_string(),
                properties: {
                    let mut props = HashMap::new();
                    props.insert("location".to_string(), Property {
                        prop_type: "string".to_string(),
                        description: Some("The city and state".to_string()),
                        items: None,
                    });
                    props
                },
                required: vec!["location".to_string()],
            },
        },
    }
}

// Register the tool with the graph
let weather_tool = create_weather_tool();
graph.register_tool(weather_tool, |args| {
    let location = args["location"].as_str().unwrap();
    Ok(serde_json::json!({
        "location": location,
        "temperature": 72,
        "condition": "Sunny"
    }))
});
```

### Creating an LLM-Powered Agent

```rust
use llmgraph::generate::generate::generate_full_response;
use llmgraph::models::tools::Message;

pub struct LLMAgent {
    api_key: String,
    model: String,
}

#[async_trait]
impl Agent for LLMAgent {
    async fn run(
        &mut self,
        input: &str,
        tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
        let tools = tool_registry.get_tools();
        
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Some("You are a helpful assistant.".to_string()),
                tool_calls: None,
            },
            Message {
                role: "user".to_string(),
                content: Some(input.to_string()),
                tool_calls: None,
            }
        ];
        
        let response = generate_full_response(
            "https://openrouter.ai/api/v1/chat/completions".to_string(),
            self.api_key.clone(),
            self.model.clone(),
            0.1,
            messages,
            Some(tools)
        ).await.unwrap();
        
        // Process tool calls if any
        if let Some(tool_calls) = &response.choices[0].message.tool_calls {
            for tool_call in tool_calls {
                let result = tool_registry.execute_tool(
                    &tool_call.function.name,
                    &tool_call.function.arguments,
                );
                // Handle tool results...
            }
        }
        
        let content = response.choices[0].message.content.clone()
            .unwrap_or_else(|| "No response".to_string());
        
        (content, Some(1)) // Route to next agent (ID: 1)
    }
    
    fn get_name(&self) -> &str {
        "LLMAgent"
    }
}
```

## Advanced Usage

### Node-Specific Tools

You can register tools for specific nodes only:

```rust
// Register a tool only for node 0
graph.register_tool_for_node(0, calculator_tool, |args| {
    // Calculator implementation
    Ok(serde_json::json!({"result": 42}))
}).unwrap();
```

### Complex Agent Chains

```rust
pub struct RouterAgent;

#[async_trait]
impl Agent for RouterAgent {
    async fn run(
        &mut self,
        input: &str,
        _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
        // Route based on input content
        if input.contains("technical") {
            ("Routing to technical team".to_string(), Some(1))
        } else if input.contains("sales") {
            ("Routing to sales team".to_string(), Some(2))
        } else {
            ("Routing to general support".to_string(), Some(3))
        }
    }
    
    fn get_name(&self) -> &str {
        "Router"
    }
}
```

### Debugging and Visualization

```rust
// Print the graph structure
graph.print();
// Output:
// Adjacency list for the Graph:
// 0 (Agent: Manager) -> 1 2
// 1 (Agent: Developer) -> 0
// 2 (Agent: Reviewer) -> 0
```

## API Reference

### Graph Methods

- `new()` - Create a new empty graph
- `add_node(id, agent)` - Add an agent node to the graph
- `add_edge(from, to)` - Connect two nodes
- `register_tool(tool, function)` - Register a global tool
- `register_tool_for_node(node_id, tool, function)` - Register a node-specific tool
- `run(start_id, input)` - Execute the graph starting from a specific node
- `print()` - Display the graph structure

### Agent Trait

Implement the `Agent` trait to create custom agents:

- `run(&mut self, input, tool_registry)` - Process input and return output with optional next node
- `get_name(&self)` - Return the agent's name for identification

### Tool Registry

The tool registry provides:

- `get_tools()` - Get all available tools
- `execute_tool(name, arguments)` - Execute a tool by name

## Examples

Check the `src/lib.rs` file for complete test examples including:

- Basic agent chains
- LLM integration with OpenRouter
- Tool calling with weather and calculator functions
- Multi-agent workflows

## Architecture

```
┌─────────────┐
│   Graph     │
├─────────────┤
│ ┌─────────┐ │     ┌──────────┐
│ │  Node 0 │─┼────▶│  Agent   │
│ └─────────┘ │     └──────────┘
│ ┌─────────┐ │     ┌──────────┐
│ │  Node 1 │─┼────▶│  Agent   │
│ └─────────┘ │     └──────────┘
│             │
│ Tool Registry│
└─────────────┘
```

## Requirements

- Rust 1.70+
- Tokio runtime for async execution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.


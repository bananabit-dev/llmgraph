//! LLMGraph - A Multi-Agent AI System Framework
//!
//! This library provides a framework for building conversational AI applications
//! with function calling capabilities using a graph-based architecture.

pub mod generate;
pub mod models;
pub mod errors;
pub mod agents;

// Re-export commonly used types for convenience
pub use errors::{LLMGraphError, LLMGraphResult};
pub use models::graph::{Agent, Graph};
pub use models::tools::{Tool, ToolRegistry, ToolRegistryTrait, Message};

// Include comprehensive test module
#[cfg(test)]
mod tests;

// Legacy function kept for compatibility
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

// ================================
// EXISTING INTEGRATION TESTS
// ================================
// These tests demonstrate end-to-end functionality with real LLM integration

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::models::graph::{Agent, Graph};
    use crate::models::tools::{
        Function, Message, Parameters, Property, Tool, ToolRegistryTrait
    };
    use async_trait::async_trait;
    use std::collections::HashMap;

    #[tokio::test]
    async fn graph_test() {
        pub struct ManagerAgent;

        #[async_trait]
        impl Agent for ManagerAgent {
            async fn run(
                &mut self,
                input: &str,
                _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
            ) -> (String, Option<i32>) {
                let response = format!("Manager received: '{}'. Delegating to developer.", input);
                (response, Some(1)) // Pass to developer (ID 1)
            }

            fn get_name(&self) -> &str {
                "Manager"
            }
        }

        pub struct DeveloperAgent;

        #[async_trait]
        impl Agent for DeveloperAgent {
            async fn run(
                &mut self,
                input: &str,
                _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
            ) -> (String, Option<i32>) {
                let response = format!(
                    "Developer working on: '{}'. Implementation complete.",
                    input
                );
                (response, None) // End of chain
            }

            fn get_name(&self) -> &str {
                "Developer"
            }
        }

        // Create a new graph
        let mut graph = Graph::new();

        // Add agents to the graph
        graph.add_node(0, Box::new(ManagerAgent));
        graph.add_node(1, Box::new(DeveloperAgent));

        // Connect the agents
        if let Err(e) = graph.add_edge(0, 1) {
            println!("Error adding edge: {}", e);
        }

        // Print the graph structure
        graph.print();

        // Run the graph starting from the manager
        println!("\nRunning the graph:");
        let output = graph.run(0, "Create a new feature").await;
        println!("{}", output);
    }

    #[tokio::test]
    async fn test_chain() {
        pub struct ManagerAgent;

        #[async_trait]
        impl Agent for ManagerAgent {
            async fn run(
                &mut self,
                input: &str,
                _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
            ) -> (String, Option<i32>) {
                let api_key =
                    "*"
                        .to_string();
                let base_url = "https://openrouter.ai/api/v1/chat/completions".to_string();
                let model = "z-ai/glm-4.5".to_string();
                let temperature = 0.1;
                let messages: Vec<Message> = vec![Message {
                    role: "system".to_string(),
                    content: Some(input.to_string()),
                    tool_calls: None,
                }];

                // Generate the response using the updated generate function
                let generated_response =
                    generate::generate::generate(base_url, api_key, model, temperature, messages)
                        .await
                        .unwrap_or_else(|_| "Failed to generate response".to_string());

                let response = format!(
                    "Manager received: '{}'. Response: '{}'. Delegating to developer.",
                    input, generated_response
                );
                (response, Some(1)) // Pass to developer (ID 1)
            }

            fn get_name(&self) -> &str {
                "Manager"
            }
        }

        pub struct DeveloperAgent;

        #[async_trait]
        impl Agent for DeveloperAgent {
            async fn run(
                &mut self,
                input: &str,
                _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
            ) -> (String, Option<i32>) {
                let response = format!(
                    "Developer working on: '{}'. Implementation complete.",
                    input
                );
                (response, None) // End of chain
            }

            fn get_name(&self) -> &str {
                "Developer"
            }
        }

        // Create a new graph
        let mut graph = Graph::new();

        // Add agents to the graph
        graph.add_node(0, Box::new(ManagerAgent));
        graph.add_node(1, Box::new(DeveloperAgent));

        // Connect the agents
        if let Err(e) = graph.add_edge(0, 1) {
            println!("Error adding edge: {}", e);
        }

        // Print the graph structure
        graph.print();

        // Run the graph starting from the manager
        println!("\nRunning the graph:");
        let output = graph
            .run(
                0,
                "what tools do you have list them and use the weather tool afterwards!",
            )
            .await;
        println!("{}", output);
    }

  #[tokio::test]
async fn test_generate_with_tools() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    
    // Create flags to track if tools were called
    let weather_called = Arc::new(AtomicBool::new(false));
    let calculator_called = Arc::new(AtomicBool::new(false));
    
    // Clone the Arcs for the closures
    let weather_called_clone = weather_called.clone();
    let calculator_called_clone = calculator_called.clone();
    
    // Define weather tool creation function
    let create_weather_tool = || -> Tool {
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
                            description: Some("The city and state, e.g. San Francisco, CA".to_string()),
                            items: None,
                        });
                        props
                    },
                    required: vec!["location".to_string()],
                },
            },
        }
    };
    
    // Define weather tool function
    let weather_tool_function = move |args: serde_json::Value| -> Result<serde_json::Value, String> {
        weather_called_clone.store(true, Ordering::SeqCst);
        let location = args["location"].as_str()
            .ok_or("Missing 'location' parameter")?;
        
        // Mock implementation
        Ok(serde_json::json!({
            "location": location,
            "temperature": 72,
            "condition": "Sunny",
            "humidity": 65
        }))
    };
    
    // Define calculator tool creation function
    let create_calculator_tool = || -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calculate".to_string(),
                description: "Perform a mathematical calculation".to_string(),
                parameters: Parameters {
                    param_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("expression".to_string(), Property {
                            prop_type: "string".to_string(),
                            description: Some("The mathematical expression to evaluate, e.g. '2 + 2'".to_string()),
                            items: None,
                        });
                        props
                    },
                    required: vec!["expression".to_string()],
                },
            },
        }
    };
    
    // Define calculator tool function
    let calculator_tool_function = move |args: serde_json::Value| -> Result<serde_json::Value, String> {
        calculator_called_clone.store(true, Ordering::SeqCst);
        let expression = args["expression"].as_str()
            .ok_or("Missing 'expression' parameter")?;
        
        // Simple calculator implementation
        let tokens: Vec<&str> = expression.split_whitespace().collect();
        if tokens.len() != 3 {
            return Err("Expression must have exactly 3 parts: num1 operator num2".to_string());
        }

        let num1: f64 = tokens[0].parse().map_err(|_| "Invalid first number")?;
        let operator = tokens[1];
        let num2: f64 = tokens[2].parse().map_err(|_| "Invalid second number")?;

        let result = match operator {
            "+" => num1 + num2,
            "-" => num1 - num2,
            "*" => num1 * num2,
            "/" => {
                if num2 == 0.0 {
                    return Err("Division by zero".to_string());
                }
                num1 / num2
            }
            _ => return Err(format!("Unknown operator: {}", operator)),
        };

        Ok(serde_json::json!({
            "expression": expression,
            "result": result
        }))
    };

    // Define a ManagerAgent that properly handles tool calling
    pub struct ManagerAgent {
        api_key: String,
        model: String,
    }

    impl ManagerAgent {
        pub fn new(api_key: String, model: String) -> Self {
            Self { api_key, model }
        }
    }

#[async_trait]
impl Agent for ManagerAgent {
    async fn run(
        &mut self,
        input: &str,
        tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
            // Get tools from the registry
            let tools = tool_registry.get_tools();
            
            // Create initial messages
            let mut messages = vec![
                Message {
                    role: "system".to_string(),
                    content: Some("You are a helpful assistant. Use the available tools to answer questions.".to_string()),
                    tool_calls: None,
                },
                Message {
                    role: "user".to_string(),
                    content: Some(input.to_string()),
                    tool_calls: None,
                }
            ];
            
            // Maximum iterations to prevent infinite loops
            let max_iterations = 5;
            let mut iteration = 0;
            
            // Agentic loop for tool calling
            while iteration < max_iterations {
                iteration += 1;
                println!("Iteration {}", iteration);
                
                // Step 1: Send request with tools
                let response = generate::generate::generate_full_response(
                    "https://openrouter.ai/api/v1/chat/completions".to_string(),
                    self.api_key.clone(),
                    self.model.clone(),
                    0.1,
                    messages.clone(),
                    Some(tools.clone())
                ).await;
                
                match response {
                    Ok(llm_response) => {
                        let choice = &llm_response.choices[0];
                        let assistant_message = &choice.message;
                        
                        // Add assistant message to conversation
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: assistant_message.content.clone(),
                            tool_calls: assistant_message.tool_calls.clone(),
                        });
                        
                        // Check if there are tool calls
                        if let Some(tool_calls) = &assistant_message.tool_calls {
                            println!("LLM requested tool calls:");
                            
                            // Step 2: Execute tools and prepare results
                            for tool_call in tool_calls {
                                println!("- Tool: {}, Arguments: {}", tool_call.function.name, tool_call.function.arguments);
                                
                                // Execute the tool
                                let tool_result = tool_registry.execute_tool(
                                    &tool_call.function.name,
                                    &tool_call.function.arguments,
                                );
                                
                                let result_content = match tool_result {
                                    Ok(result) => {
                                        let result_str = serde_json::to_string(&result)
                                            .unwrap_or_else(|_| "Tool result".to_string());
                                        println!("Tool result: {}", result_str);
                                        result_str
                                    }
                                    Err(e) => {
                                        println!("Tool error: {}", e);
                                        format!("Error: {}", e)
                                    }
                                };
                                
                                // Step 3: Add tool result to conversation
                                messages.push(Message {
                                    role: "tool".to_string(),
                                    content: Some(result_content),
                                    tool_calls: None,
                                });
                            }
                            
                            // Continue the loop to let the model respond to tool results
                            continue;
                        } else if let Some(content) = &assistant_message.content {
                            println!("LLM responded without tool calls: {}", content);
                            
                            // No tool calls, we're done
                            return (format!("Manager processed: '{}'. Result: {}", input, content), Some(1));
                        } else {
                            return (format!("Manager received empty response for: '{}'", input), Some(1));
                        }
                    }
                    Err(e) => {
                        println!("Error generating response: {}", e);
                        return (format!("Error processing request: {}", e), Some(1));
                    }
                }
            }
            
            // If we reached max iterations, return the last response
            if let Some(last_message) = messages.last() {
                if let Some(content) = &last_message.content {
                    return (format!("Manager processed: '{}'. Result: {}", input, content), Some(1));
                }
            }
            
            (format!("Manager reached max iterations for: '{}'", input), Some(1))
        }

        fn get_name(&self) -> &str {
            "ManagerAgent"
        }
    }

    // Define a DeveloperAgent
    pub struct DeveloperAgent;

    #[async_trait]
    impl Agent for DeveloperAgent {
        async fn run(&mut self, input: &str, _tool_registry: &(dyn ToolRegistryTrait + Send + Sync)) -> (String, Option<i32>) {
            let response = format!("Developer received: '{}'. Processing complete.", input);
            (response, None) // End of chain
        }

        fn get_name(&self) -> &str {
            "DeveloperAgent"
        }
    }
    
    // Create a new graph
    let mut graph = Graph::new();
    
    // Register tools with the mock functions
    graph.register_tool(create_weather_tool(), weather_tool_function);
    graph.register_tool(create_calculator_tool(), calculator_tool_function);
    
    // Create agents
    let manager_agent = ManagerAgent::new(
        "{{api_key}}".to_string(), // Replace with your actual API key
        "model".to_string(),
    );
    
    // Add agents to the graph
    graph.add_node(0, Box::new(manager_agent));
    graph.add_node(1, Box::new(DeveloperAgent));
    
    // Connect the agents
    if let Err(e) = graph.add_edge(0, 1) {
        println!("Error adding edge: {}", e);
    }
    
    // Print the graph structure
    graph.print();
    
    // Run the graph with a prompt that requires tool usage
    println!("\nRunning the graph with tool calling:");
    let output = graph.run(0, "What's the weather in Boston and calculate 20 * 5?").await;
    println!("{}", output);
    
    // Assert that the output contains expected information
    assert!(output.contains("Boston") || output.contains("weather"));
    assert!(output.contains("20 * 5") || output.contains("100"));
    assert!(output.contains("Manager processed"));
    assert!(output.contains("Developer received"));
    
    // Check if the tools were called
    assert!(weather_called.load(Ordering::SeqCst), "Weather tool was not called");
    assert!(calculator_called.load(Ordering::SeqCst), "Calculator tool was not called");
    
    println!("✓ Weather tool was called: {}", weather_called.load(Ordering::SeqCst));
    println!("✓ Calculator tool was called: {}", calculator_called.load(Ordering::SeqCst));
}
}
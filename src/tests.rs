//! Comprehensive test suite for the LLMGraph library
//! 
//! This module contains tests for:
//! - Core graph functionality
//! - Built-in agent types
//! - Tool registration and execution
//! - Error handling
//! - State persistence
//! - API integration

#[cfg(test)]
mod tests {
    use crate::agents::*;
    use crate::agents::router::RouteRule;
    use crate::agents::retry::RetryStrategy;
    use crate::agents::summarizer::{SummarizerConfig, SummaryStyle};
    use crate::models::graph::{Agent, Graph};
    use crate::models::tools::{
        Function, Message, Parameters, Property, Tool, ToolRegistryTrait
    };
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::time::Duration;

    // ================================
    // GLOBAL TEST CONFIGURATION
    // ================================
    
    /// Global API key for testing - REPLACE THIS with a new key after testing
    const TEST_API_KEY: &str = "{{load from dotenv}}";
    const TEST_MODEL: &str = "{{load model}}";
    const TEST_BASE_URL: &str = "{{api_base_url}}";

    // ================================
    // TEST AGENTS
    // ================================

    /// Simple test agent that echoes input
    pub struct EchoAgent {
        name: String,
    }

    impl EchoAgent {
        pub fn new(name: impl Into<String>) -> Self {
            Self { name: name.into() }
        }
    }

    #[async_trait]
    impl Agent for EchoAgent {
        async fn run(
            &mut self,
            input: &str,
            _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
        ) -> (String, Option<i32>) {
            (format!("Echo: {}", input), None)
        }

        fn get_name(&self) -> &str {
            &self.name
        }
    }

    /// Test agent that always routes to a specific node
    pub struct FixedRouteAgent {
        name: String,
        target_node: i32,
    }

    impl FixedRouteAgent {
        pub fn new(name: impl Into<String>, target: i32) -> Self {
            Self {
                name: name.into(),
                target_node: target,
            }
        }
    }

    #[async_trait]
    impl Agent for FixedRouteAgent {
        async fn run(
            &mut self,
            input: &str,
            _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
        ) -> (String, Option<i32>) {
            (
                format!("Routing '{}' to node {}", input, self.target_node),
                Some(self.target_node),
            )
        }

        fn get_name(&self) -> &str {
            &self.name
        }
    }

    // ================================
    // CORE GRAPH TESTS
    // ================================

    /// Test basic graph construction and node addition
    #[tokio::test]
    async fn test_graph_construction() {
        let mut graph = Graph::new();
        
        // Add nodes
        graph.add_node(0, Box::new(EchoAgent::new("Agent0")));
        graph.add_node(1, Box::new(EchoAgent::new("Agent1")));
        
        // Add edge
        let result = graph.add_edge(0, 1);
        assert!(result.is_ok(), "Failed to add edge between existing nodes");
        
        // Try to add edge with non-existent node
        let result = graph.add_edge(0, 99);
        assert!(result.is_err(), "Should fail when adding edge to non-existent node");
    }

    /// Test graph execution with simple chain
    #[tokio::test]
    async fn test_simple_chain_execution() {
        let mut graph = Graph::new();
        
        // Create a simple chain: Agent0 -> Agent1 -> Agent2
        graph.add_node(0, Box::new(FixedRouteAgent::new("Router", 1)));
        graph.add_node(1, Box::new(FixedRouteAgent::new("Processor", 2)));
        graph.add_node(2, Box::new(EchoAgent::new("Final")));
        
        // Connect nodes
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();
        
        // Execute
        let result = graph.run(0, "test input").await;
        
        assert!(result.contains("Routing"));
        assert!(result.contains("Echo"));
        println!("Chain execution result:\n{}", result);
    }

    // ================================
    // ROUTER AGENT TESTS
    // ================================

    /// Test router agent with pattern matching
    #[tokio::test]
    #[allow(unused_variables)]
    async fn test_router_agent() {
        let mut router = RouterAgent::new()
            .add_route(RouteRule {
                pattern: r"technical|bug|error".to_string(),
                target_node: 1,
                description: Some("Technical support".to_string()),
            })
            .add_route(RouteRule {
                pattern: r"sales|pricing|buy".to_string(),
                target_node: 2,
                description: Some("Sales team".to_string()),
            })
            .set_default(3);

        // Test technical routing
        let (response, next) = router.run("I found a bug", &EmptyRegistry).await;
        assert_eq!(next, Some(1), "Should route to technical support");
        
        // Test sales routing
        let (response, next) = router.run("What's the pricing?", &EmptyRegistry).await;
        assert_eq!(next, Some(2), "Should route to sales");
        
        // Test default routing
        let (response, next) = router.run("General question", &EmptyRegistry).await;
        assert_eq!(next, Some(3), "Should route to default");
        
        println!("Router test passed");
    }

    // ================================
    // STATEFUL AGENT TESTS
    // ================================

    /// Test stateful agent with persistence
    #[tokio::test]
    async fn test_stateful_agent() {
        let mut agent = StatefulAgent::new("TestStateful")
            .with_max_history(5);

        // First execution
        let (response1, _) = agent.run("First input", &EmptyRegistry).await;
        assert!(response1.contains("execution #1"));

        // Second execution
        let (response2, _) = agent.run("Second input", &EmptyRegistry).await;
        assert!(response2.contains("execution #2"));

        // Test state persistence
        let state = agent.get_state();
        assert_eq!(state.execution_count, 2);
        assert_eq!(state.history.len(), 2);
        
        // Test state serialization
        let json = agent.save_state().unwrap();
        println!("Serialized state: {}", json);
        
        // Create new agent and load state
        let mut new_agent = StatefulAgent::new("NewAgent");
        new_agent.load_state(&json).unwrap();
        
        let loaded_state = new_agent.get_state();
        assert_eq!(loaded_state.execution_count, 2);
        assert_eq!(loaded_state.history.len(), 2);
        
        println!("Stateful agent test passed");
    }

    // ================================
    // RETRY AGENT TESTS
    // ================================

    /// Test agent that fails a certain number of times
    pub struct FlakeyAgent {
        failures_remaining: std::sync::Mutex<usize>,
    }

    impl FlakeyAgent {
        pub fn new(failures: usize) -> Self {
            Self {
                failures_remaining: std::sync::Mutex::new(failures),
            }
        }
    }

    #[async_trait]
    impl Agent for FlakeyAgent {
        async fn run(
            &mut self,
            input: &str,
            _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
        ) -> (String, Option<i32>) {
            let mut failures = self.failures_remaining.lock().unwrap();
            if *failures > 0 {
                *failures -= 1;
                ("Error: temporary failure".to_string(), None)
            } else {
                (format!("Success: {}", input), None)
            }
        }

        fn get_name(&self) -> &str {
            "FlakeyAgent"
        }
    }

    /// Test retry agent with exponential backoff
    #[tokio::test]
    async fn test_retry_agent() {
        let flakey = Box::new(FlakeyAgent::new(2)); // Fail twice, then succeed
        
        let mut retry_agent = RetryAgent::new(flakey)
            .with_max_retries(3)
            .with_strategy(RetryStrategy::Fixed(Duration::from_millis(10)))
            .with_verbose(true);

        let (response, _) = retry_agent.run("test", &EmptyRegistry).await;
        assert!(response.contains("Success"), "Should eventually succeed");
        
        println!("Retry agent test passed");
    }

    // ================================
    // VALIDATOR AGENT TESTS
    // ================================

    /// Test validator agent with multiple rules
    #[tokio::test]
    #[allow(unused_variables)]
    async fn test_validator_agent() {
        let mut validator = ValidatorAgent::new()
            .add_length_rule(Some(5), Some(100), true)
            .add_pattern_rule(
                "no_numbers",
                r"^[^0-9]+$",
                "Input should not contain numbers",
                false,
            )
            .with_success_route(1)
            .with_failure_route(2);

        // Test valid input
        let (response, next) = validator.run("Hello World", &EmptyRegistry).await;
        assert_eq!(next, Some(1), "Valid input should route to success node");
        
        // Test too short input
        let (response, next) = validator.run("Hi", &EmptyRegistry).await;
        assert_eq!(next, Some(2), "Too short input should route to failure node");
        assert!(response.contains("too short"));
        
        // Test input with numbers (warning only)
        let (response, next) = validator.run("Hello123", &EmptyRegistry).await;
        assert_eq!(next, Some(1), "Should pass with warning");
        assert!(response.contains("Warning"));
        
        println!("Validator agent test passed");
    }

    // ================================
    // SUMMARIZER AGENT TESTS
    // ================================

    /// Test summarizer agent with different styles
    #[tokio::test]
    async fn test_summarizer_agent() {
        let mut summarizer = SummarizerAgent::new()
            .with_config(SummarizerConfig {
                max_length: 50,
                style: SummaryStyle::Bullets,
                ..Default::default()
            });

        let long_text = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence.";
        
        let (response, _) = summarizer.run(long_text, &EmptyRegistry).await;
        assert!(response.contains("•"), "Should format as bullet points");
        
        println!("Summarizer test: {}", response);
    }

    // ================================
    // TOOL REGISTRY TESTS
    // ================================

    /// Test tool registration and execution
    #[tokio::test]
    async fn test_tool_registry() {
        let mut graph = Graph::new();
        
        // Create a calculator tool
        let calc_tool = Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calculator".to_string(),
                description: "Performs basic math".to_string(),
                parameters: Parameters {
                    param_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("operation".to_string(), Property {
                            prop_type: "string".to_string(),
                            description: Some("add, subtract, multiply, divide".to_string()),
                            items: None,
                        });
                        props.insert("a".to_string(), Property {
                            prop_type: "number".to_string(),
                            description: Some("First number".to_string()),
                            items: None,
                        });
                        props.insert("b".to_string(), Property {
                            prop_type: "number".to_string(),
                            description: Some("Second number".to_string()),
                            items: None,
                        });
                        props
                    },
                    required: vec!["operation".to_string(), "a".to_string(), "b".to_string()],
                },
            },
        };
        
        // Register the tool
        graph.register_tool(calc_tool, |args| {
            let op = args["operation"].as_str().unwrap_or("");
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            
            let result = match op {
                "add" => a + b,
                "subtract" => a - b,
                "multiply" => a * b,
                "divide" => if b != 0.0 { a / b } else { 0.0 },
                _ => 0.0,
            };
            
            Ok(serde_json::json!({ "result": result }))
        });
        
        // Test tool execution
        let registry = graph.get_shared_tool_registry();
        let tools = registry.get_tools();
        assert_eq!(tools.len(), 1, "Should have one tool registered");
        
        let result = registry.execute_tool(
            "calculator",
            r#"{"operation": "add", "a": 5, "b": 3}"#,
        );
        
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["result"], 8.0);
        
        println!("Tool registry test passed");
    }

    // ================================
    // ERROR HANDLING TESTS
    // ================================

    /// Test error types and conversion
    #[test]
    fn test_error_handling() {
        use crate::errors::*;
        
        // Test NodeError
        let node_err = NodeError::NodeNotFound(42);
        let llm_err: LLMGraphError = node_err.into();
        assert!(format!("{}", llm_err).contains("Node 42 not found"));
        
        // Test ToolError
        let tool_err = ToolError::ToolNotFound("my_tool".to_string());
        let llm_err: LLMGraphError = tool_err.into();
        assert!(format!("{}", llm_err).contains("Tool 'my_tool' not found"));
        
        // Test ApiError
        let api_err = ApiError::RateLimitExceeded;
        let llm_err: LLMGraphError = api_err.into();
        assert!(format!("{}", llm_err).contains("rate limit"));
        
        println!("Error handling test passed");
    }

    // ================================
    // INTEGRATION TEST WITH LLM
    // ================================

    /// Integration test with actual LLM API (requires valid API key)
    #[tokio::test]
    #[ignore] // Remove ignore to run with actual API
    async fn test_llm_integration() {
        use crate::generate::generate::generate;
        
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Some("You are a helpful assistant.".to_string()),
                tool_calls: None,
            },
            Message {
                role: "user".to_string(),
                content: Some("Say 'test successful' if you can read this.".to_string()),
                tool_calls: None,
            },
        ];
        
        let result = generate(
            TEST_BASE_URL.to_string(),
            TEST_API_KEY.to_string(),
            TEST_MODEL.to_string(),
            0.1,
            messages,
        ).await;
        
        assert!(result.is_ok(), "API call should succeed");
        let response = result.unwrap();
        println!("LLM Response: {}", response);
    }

    // ================================
    // HELPER STRUCTURES
    // ================================

    /// Empty tool registry for testing
    struct EmptyRegistry;
    
    impl ToolRegistryTrait for EmptyRegistry {
        fn get_tools(&self) -> Vec<Tool> {
            Vec::new()
        }
        
        fn execute_tool(&self, _name: &str, _arguments: &str) -> Result<serde_json::Value, String> {
            Err("No tools available".to_string())
        }
    }
}
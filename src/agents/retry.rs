//! Retry agent for handling failures with configurable retry strategies.

use async_trait::async_trait;
use crate::models::graph::Agent;
use crate::models::tools::ToolRegistryTrait;
use std::time::Duration;
use tokio::time::sleep;

/// Retry strategy configuration
#[derive(Debug, Clone)]
pub enum RetryStrategy {
    /// Fixed delay between retries
    Fixed(Duration),
    /// Exponential backoff with base delay
    ExponentialBackoff {
        base: Duration,
        max: Duration,
        multiplier: f64,
    },
    /// Linear backoff with increment
    Linear {
        initial: Duration,
        increment: Duration,
        max: Duration,
    },
}

/// Retry agent that wraps another agent and retries on failure.
///
/// # Example
/// ```rust,ignore
/// use llmgraph::agents::RetryAgent;
/// use llmgraph::agents::retry::RetryStrategy;
/// use std::time::Duration;
///
/// // Assuming inner_agent is defined
/// let retry_agent = RetryAgent::new(inner_agent)
///     .with_max_retries(3)
///     .with_strategy(RetryStrategy::ExponentialBackoff {
///         base: Duration::from_millis(100),
///         max: Duration::from_secs(10),
///         multiplier: 2.0,
///     })
///     .with_retry_condition(|output| {
///         // Retry if output contains "error"
///         output.contains("error")
///     });
/// ```
pub struct RetryAgent {
    inner: Box<dyn Agent>,
    max_retries: usize,
    strategy: RetryStrategy,
    retry_condition: Option<Box<dyn Fn(&str) -> bool + Send + Sync>>,
    name: String,
    verbose: bool,
}

impl RetryAgent {
    /// Create a new retry agent wrapping an inner agent
    pub fn new(inner: Box<dyn Agent>) -> Self {
        let inner_name = inner.get_name().to_string();
        Self {
            inner,
            max_retries: 3,
            strategy: RetryStrategy::Fixed(Duration::from_secs(1)),
            retry_condition: None,
            name: format!("Retry[{}]", inner_name),
            verbose: false,
        }
    }

    /// Set maximum number of retries
    pub fn with_max_retries(mut self, max: usize) -> Self {
        self.max_retries = max;
        self
    }

    /// Set the retry strategy
    pub fn with_strategy(mut self, strategy: RetryStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set a custom retry condition
    pub fn with_retry_condition<F>(mut self, condition: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.retry_condition = Some(Box::new(condition));
        self
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Calculate delay for a given retry attempt
    fn calculate_delay(&self, attempt: usize) -> Duration {
        match &self.strategy {
            RetryStrategy::Fixed(delay) => *delay,
            RetryStrategy::ExponentialBackoff { base, max, multiplier } => {
                let delay = base.as_millis() as f64 * multiplier.powi(attempt as i32);
                let delay_ms = delay.min(max.as_millis() as f64) as u64;
                Duration::from_millis(delay_ms)
            }
            RetryStrategy::Linear { initial, increment, max } => {
                let delay = initial.as_millis() + (increment.as_millis() * attempt as u128);
                let delay_ms = delay.min(max.as_millis()) as u64;
                Duration::from_millis(delay_ms)
            }
        }
    }

    /// Check if output indicates a failure that should be retried
    fn should_retry(&self, output: &str) -> bool {
        if let Some(condition) = &self.retry_condition {
            condition(output)
        } else {
            // Default: retry on empty output or error keywords
            output.is_empty() 
                || output.to_lowercase().contains("error")
                || output.to_lowercase().contains("failed")
                || output.to_lowercase().contains("exception")
        }
    }
}

#[async_trait]
impl Agent for RetryAgent {
    async fn run(
        &mut self,
        input: &str,
        tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
        let mut last_output = String::new();
        let mut last_next_node = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay = self.calculate_delay(attempt - 1);
                if self.verbose {
                    println!("Retry attempt {} after {:?} delay", attempt, delay);
                }
                sleep(delay).await;
            }

            let (output, next_node) = self.inner.run(input, tool_registry).await;
            
            if !self.should_retry(&output) || attempt == self.max_retries {
                if self.verbose && attempt > 0 {
                    println!("Success after {} retries", attempt);
                }
                return (output, next_node);
            }

            if self.verbose {
                println!("Attempt {} failed: {}", attempt + 1, output);
            }

            last_output = output;
            last_next_node = next_node;
        }

        // Return last attempt's result if all retries exhausted
        let failure_msg = format!(
            "All {} retries exhausted. Last output: {}",
            self.max_retries, last_output
        );
        (failure_msg, last_next_node)
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}
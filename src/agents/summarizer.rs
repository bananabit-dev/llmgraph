//! Summarizer agent for condensing long texts or conversation histories.

use async_trait::async_trait;
use crate::models::graph::Agent;
use crate::models::tools::ToolRegistryTrait;

/// Configuration for summarization
#[derive(Debug, Clone)]
pub struct SummarizerConfig {
    /// Maximum length of summary in characters
    pub max_length: usize,
    /// Whether to preserve key entities
    pub preserve_entities: bool,
    /// Whether to include timestamps
    pub include_timestamps: bool,
    /// Summary style
    pub style: SummaryStyle,
}

#[derive(Debug, Clone)]
pub enum SummaryStyle {
    /// Bullet points
    Bullets,
    /// Paragraph form
    Paragraph,
    /// Key points only
    KeyPoints,
    /// Executive summary
    Executive,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            max_length: 500,
            preserve_entities: true,
            include_timestamps: false,
            style: SummaryStyle::Paragraph,
        }
    }
}

/// Summarizer agent that condenses input text.
/// 
/// # Example
/// ```rust
/// use llmgraph::agents::SummarizerAgent;
/// use llmgraph::agents::summarizer::{SummarizerConfig, SummaryStyle};
/// 
/// let summarizer = SummarizerAgent::new()
///     .with_config(SummarizerConfig {
///         max_length: 200,
///         style: SummaryStyle::Bullets,
///         ..Default::default()
///     });
/// ```
pub struct SummarizerAgent {
    config: SummarizerConfig,
    name: String,
    history: Vec<String>,
    max_history: usize,
}

impl SummarizerAgent {
    /// Create a new summarizer agent
    pub fn new() -> Self {
        Self {
            config: SummarizerConfig::default(),
            name: "Summarizer".to_string(),
            history: Vec::new(),
            max_history: 10,
        }
    }

    /// Set the configuration
    pub fn with_config(mut self, config: SummarizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the agent name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Simple summarization logic (in production, use an LLM)
    fn summarize(&self, text: &str) -> String {
        let sentences: Vec<&str> = text.split(". ").collect();
        
        match self.config.style {
            SummaryStyle::Bullets => {
                let bullet_points: Vec<String> = sentences
                    .iter()
                    .take(3)
                    .map(|s| format!("• {}", s.trim()))
                    .collect();
                bullet_points.join("\n")
            }
            SummaryStyle::KeyPoints => {
                let key_points: Vec<String> = sentences
                    .iter()
                    .filter(|s| {
                        s.contains("important") || 
                        s.contains("key") || 
                        s.contains("critical") ||
                        s.contains("main")
                    })
                    .take(3)
                    .map(|s| s.to_string())
                    .collect();
                
                if key_points.is_empty() {
                    sentences.iter().take(2).map(|s| s.to_string()).collect::<Vec<_>>().join(". ")
                } else {
                    key_points.join(". ")
                }
            }
            _ => {
                // For other styles, just truncate to max_length
                if text.len() > self.config.max_length {
                    format!("{}...", &text[..self.config.max_length.min(text.len())])
                } else {
                    text.to_string()
                }
            }
        }
    }
}

impl Default for SummarizerAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for SummarizerAgent {
    async fn run(
        &mut self,
        input: &str,
        _tool_registry: &(dyn ToolRegistryTrait + Send + Sync),
    ) -> (String, Option<i32>) {
        // Add to history
        self.history.push(input.to_string());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        // Summarize the input
        let summary = self.summarize(input);
        
        let response = format!("Summary: {}", summary);
        (response, None)
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}
//! LLM API integration module for generating responses with OpenRouter.
//!
//! This module provides functions to interact with LLM APIs, supporting
//! both simple text generation and function calling capabilities.

use reqwest::{header::{HeaderMap, HeaderValue}, Client};
use serde::{Deserialize, Serialize};
use crate::models::tools::*;

#[derive(Serialize)]
struct ChatCompletion {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    tools: Option<Vec<Tool>>,
}

#[derive(Deserialize)]
struct APIResponse {
    choices: Vec<Choice>,
}

/// Generate a text response from an LLM without tool support.
///
/// # Arguments
/// * `base_url` - The API endpoint URL
/// * `api_key` - Your API key for authentication
/// * `model` - The model to use (e.g., "gpt-4", "claude-3")
/// * `temperature` - Sampling temperature (0.0 to 1.0)
/// * `messages` - The conversation history
///
/// # Returns
/// * `Ok(String)` - The generated text response
/// * `Err(reqwest::Error)` - If the API request fails
///
/// # Example
/// ```rust,ignore
/// # use llmgraph::generate::generate::generate;
/// # use llmgraph::Message;
/// # // In async context with defined variables
/// let response = generate(
///     "https://api.openrouter.ai/v1/chat/completions".to_string(),
///     api_key,
///     "gpt-4".to_string(),
///     0.7,
///     messages
/// ).await?;
/// ```
pub async fn generate(
    base_url: String,
    api_key: String,
    model: String,
    temperature: f32,
    messages: Vec<Message>
) -> Result<String, reqwest::Error> {
    generate_with_tools(base_url, api_key, model, temperature, messages, None).await
}

/// Generate a response from an LLM with optional tool/function calling support.
///
/// # Arguments
/// * `base_url` - The API endpoint URL
/// * `api_key` - Your API key for authentication
/// * `model` - The model to use
/// * `temperature` - Sampling temperature (0.0 to 1.0)
/// * `messages` - The conversation history
/// * `tools` - Optional tools available for the model to call
///
/// # Returns
/// * `Ok(String)` - The generated text response
/// * `Err(reqwest::Error)` - If the API request fails
pub async fn generate_with_tools(
    base_url: String,
    api_key: String,
    model: String,
    temperature: f32,
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>
) -> Result<String, reqwest::Error> {
    let content_type: &str = "application/json";

    let mut headers = HeaderMap::new();

    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .expect("Failed to add api key to Bearer"),
    );

    headers.insert(
        "content-type",
        HeaderValue::from_str(content_type).expect("Failed to add content-type")
    );

    let client = Client::builder().default_headers(headers).build().expect("Can't build client");

    let chat_completion = ChatCompletion {
        model,
        messages,
        temperature,
        tools,
    };

    let response = client
        .post(base_url)
        .json(&chat_completion)
        .send()
        .await?;

    let res: APIResponse = response.json().await?;

    // Extract the content from the first choice
    if let Some(choice) = res.choices.first() {
        if let Some(content) = &choice.message.content {
            return Ok(content.clone());
        }
    }

    // If there's no content, return an empty string
    Ok(String::new())
}

/// Generate a full response from an LLM including tool calls.
///
/// Unlike `generate_with_tools`, this returns the full response structure
/// including any tool calls the model wants to make.
///
/// # Arguments
/// * `base_url` - The API endpoint URL
/// * `api_key` - Your API key for authentication
/// * `model` - The model to use
/// * `temperature` - Sampling temperature (0.0 to 1.0)
/// * `messages` - The conversation history
/// * `tools` - Optional tools available for the model to call
///
/// # Returns
/// * `Ok(LLMResponse)` - The full response including tool calls
/// * `Err(reqwest::Error)` - If the API request fails
///
/// # Example
/// ```rust,ignore
/// # use llmgraph::generate::generate::generate_full_response;
/// # // In async context with defined variables
/// let response = generate_full_response(
///     base_url,
///     api_key,
///     model,
///     0.1,
///     messages,
///     Some(tools)
/// ).await?;
///
/// if let Some(tool_calls) = &response.choices[0].message.tool_calls {
///     for tool_call in tool_calls {
///         // Handle tool calls
///     }
/// }
/// ```
pub async fn generate_full_response(
    base_url: String,
    api_key: String,
    model: String,
    temperature: f32,
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>
) -> Result<LLMResponse, reqwest::Error> {
    let content_type: &str = "application/json";

    let mut headers = HeaderMap::new();

    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .expect("Failed to add api key to Bearer"),
    );

    headers.insert(
        "content-type",
        HeaderValue::from_str(content_type).expect("Failed to add content-type")
    );

    let client = Client::builder().default_headers(headers).build().expect("Can't build client");

    let chat_completion = ChatCompletion {
        model,
        messages,
        temperature,
        tools,
    };

    let response = client
        .post(base_url)
        .json(&chat_completion)
        .send()
        .await?;

    let res: LLMResponse = response.json().await?;
    Ok(res)
}
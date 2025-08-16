//! Built-in agent implementations for common use cases.

pub mod router;
pub mod summarizer;
pub mod validator;
pub mod retry;
pub mod parallel;
pub mod state;

pub use router::RouterAgent;
pub use summarizer::SummarizerAgent;
pub use validator::ValidatorAgent;
pub use retry::RetryAgent;
pub use parallel::ParallelAgent;
pub use state::StatefulAgent;
#[derive(Debug, thiserror::Error)]
pub enum ConfigError{
    #[error("invalid config file:{0}")]
    Validationerror(String),

    #[error("failed to read config file:{0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse config file:{0}")]
    JsonError(#[from] serde_json::Error),
}
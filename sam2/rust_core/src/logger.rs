use anyhow::Result;
use std::path::PathBuf;
use tracing::{info, warn, error};
use tracing_subscriber::fmt;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::time::LocalTime;

#[derive(Debug)]
pub struct LoggerConfig {
    pub log_dir: String,
    pub log_level: String,
    pub max_size: i32,
    pub max_files: i32,
    pub log_file: String,
}

impl LoggerConfig {
    pub fn new() -> Self {
        Self { 
            log_dir:String::new(),
            log_level: "INFO".to_string(), 
            max_size: 10, 
            max_files: 5,
            log_file: String::new(),
        }
    }
}

pub struct Logger {
    pub config: LoggerConfig,
}

impl Logger {
    pub fn new(config: LoggerConfig) ->Result<Self> {
        let logger = Logger { config };
        logger.init()?;
        Ok(logger)
    }

    fn init(&self) -> Result<()> {
        std::fs::create_dir_all(&self.config.log_dir)?;

        let file_appender = RollingFileAppender::new(
            Rotation::new(
                PathBuf::from(&self.config.log_dir), // log directory
                "log.log", // log file name
                self.config.max_size * 1024 * 1024, // convert MB to bytes  
                self.config.max_files, // max number of log files
            )
        );
        let subscriber = fmt::Subscriber::builder()
            .with_ansi(false)
            .with_timer(LocalTime::rfc_3339())
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_writer(file_appender)
            .with_env_filter(EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&self.config.log_level))
            )
            .build();

        tracing::subscriber::set_global_default(subscriber)
            .map_err(|e| anyhow::anyhow!("Failed to set global default subscriber: {}", e))?;

        info!("Logger initialized successfully");
        Ok(())
    }

    fn info(&self, message: &str) { // rust function definition
        info!("{}", message);
    }
    
    fn warn(&self, message: &str) {
        warn!("{}", message);
    }
    
    fn error(&self, message: &str) {
        error!("{}", message);
    }
}



pub fn register_logger(py:Python<'_>, m:PyModule) ->PyResult<()> {
    m.add_class::<LoggerConfig>()?;
    m.add_class::<Logger>()?;
    Ok(())
}



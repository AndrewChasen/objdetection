use pyo3::prelude::*;
use tracing::{info, warn, error};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::fmt::{self, EnvFilter, LocalTime};
use std::path::PathBuf;
use anyhow::Result;

use super::config::LoggerConfig;

#[pyclass]
pub struct Logger {
    #[pyo3(get, set)]
    pub config: LoggerConfig,
}

#[pymethods]
impl Logger {
    #[new] // equivalent to __init__ in python  
    pub fn new(config: LoggerConfig) ->PyResult<Self> {
        let logger = Logger { config };
        logger.init().map_err(|e| PyErr::new::<pyo3::exceptions::PyException,_>(
            format!("Failed to initialize logger: {}", e)
        ))?;
        Ok(logger)
    }

    fn init(&self) -> Result<()> {
        std::fs::create_dir_all(&self.config.log_dir)?;

        let file_appender = RollingFileAppender::new(
            Rotation::new(
                PathBuf::from(&self.config.log_dir), // log directory
                &self.config.log_file, // log file name
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
}

#[pymethods]
impl Logger {
    #[pyo3(text_signature = "{$self, message}")] // python method signature
    fn info(&self, message: &str) { // rust function definition
        info!("{}", message);
    }

    #[pyo3(text_signature = "{$self, message}")]
    fn warn(&self, message: &str) {
        warn!("{}", message);
    }

    #[pyo3(text_signature = "{$self, message}")]
    fn error(&self, message: &str) {
        error!("{}", message);
    }
}

pub fn register_logger(_py:Python<'_>, m:&PyModule) ->PyResult<()> {
    m.add_class::<LoggerConfig>()?;
    m.add_class::<Logger>()?;
    Ok(())
}



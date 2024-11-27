#[pyclass] // rust struct can be accessed by python
#[derive(Debug)]
pub struct LoggerConfig {
    #[pyo3(get, set)] // feild can be accessed by python
    pub log_dir: String,
    #[pyo3(get, set)]
    pub log_level: String,
    #[pyo3(get, set)]
    pub max_size: u64,
    #[pyo3(get, set)]
    pub max_files: u32,
    #[pyo3(get, set)]
    pub log_file: String,
}

#[pymethods] // methods can be accessed by python
impl LoggerConfig {
    #[new] // equivalent to __init__ in python  
    pub fn new() -> Self {
        Self::default()
    }

    #[staticmethod]
    pub fn with_params(log_dir: String, log_level: String, max_size: u64, max_files: u32, log_file: String) -> Self {
        Self { log_dir, log_level, max_size, max_files, log_file }
    }
}

impl Default for LoggerConfig {
    fn default() -> Self { 
        Self { 
            log_dir: String::from("logs"),
            log_level: "INFO".to_string(), 
            max_size: 10, 
            max_files: 5,
            log_file: "log.log".to_string(),
        }
    }
}
use pyo3::prelude::*;

mod frame;
mod logging;

use frame::PyFrameConfig;

#[pymodule]
fn sam2_rust_bridge(py:Python, m:&PyModule)->PyResult<()>{
    logging::init_logging()?;

    m.add_class::<frame::PyFrameConfig>()?;
    logging::register_logger(py, m)?;
    

    Ok(())
}
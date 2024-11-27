use pyo3::prelude::*;
use pyo3::types::PyDict;
use rust_core::config::frame_config::{FrameConfig, ImageFormat};
use std::collections::HashMap;

#[pyclass(name = "FrameConfig")]
pub struct PyFrameConfig{
    inner:FrameConfig,
}

#[pymethods]
impl PyFrameConfig{
    #[new]
    pub fn new()->Self{
       Self{
        inner:FrameConfig::new(),
       }
    }

    #[staticmethod]
    pub fn load_from_file(path:String)->PyResult<Self>{
        let config = FrameConfig::load_from_file(path).map_err( |e|PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self{inner:config})
    }

    pub fn save_to_file(&self, path:String)->PyResult<()>{
        self.inner.save_to_file(path)
            .map_err(|e|PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        Ok(())
    }

    pub fn update(&mut self, update:&PyDict)->PyResult<()>{
        let mut update_map = HashMap::new();
        for (key, value) in update.iter(){
            let key = key.extract::<String>()?;
            let value = value.extract()?;
            update_map.insert(key, value);
        }
        self.inner.update(update_map).map_err(|e|PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    #[getter]
    pub fn get_output_dir(&self)->String{
        self.inner.output_dir.to_string_lossy().into_owned()
    }

    #[getter]
    pub fn get_video_path(&self)->String{
        self.inner.video_path().to_string_lossy().into_owned()
    }
}

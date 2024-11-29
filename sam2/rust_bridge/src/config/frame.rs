use serde_json::Value;
use pyo3::prelude::*;
use shared_memory::{ShmemConf,Shmem};
use numpy::{PyArray,PyArray3};
use std::sync::Arc;

#[pyclass]
pub struct PyFrameConfig{
    video_path:String,
    output_dir:String,
}

#[pymethods]
impl PyFrameConfig{
    #[new]
    pub fn new()->Self{
        Self{
            video_path:String::new(), 
            output_dir:String::new()
        }
    }

    #[pyo3(name = "process_batch_shared_memory")]
    pub fn process_batch_shared_memory(&self, metadata:String)->PyResult<()>{
        // 解析metadata, 使用json库
        let metadata:Value = serde_json::from_str(&metadata)?;

        // 获取frame_metadata，按照键值对遍历
        if let Some(frame_metadata) = metadata.get("frame_metadata").as_array(){
            // 遍历frame_metadata
            for frame_meta in frame_metadata{
                // 获取shm_name
                let shm_name = frame_meta.get("shm_name").as_str()
                    .ok_or_else(||PyErr::new::<pyo3::exceptions::PyValueError, _>("shm_name is required"))?;

                // 获取shape
                let shape:Vec<usize> = serde_json::from_value(frame_meta.get("shape").clone())
                    .map_err(|e|PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

                // 打开共享内存
               let shmem = ShmemConf::new().os_share_name(shm_name)
               .open().map_err(|e|PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

                // 拿到处理帧，然后处理了。
               self.process_frame(&shmem, shape)?;
            }
        }
        Ok(())
    }
}
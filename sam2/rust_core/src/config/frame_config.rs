use super::error::ConfigError;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::{Context, Result};
use log::{error, info, warn};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneDetectionConfig {
    pub min_scene_len: f32,
    pub threshold: f32,
}

impl SceneDetectionConfig {
    pub fn new(threshold: f32, min_scene_len: f32) -> Self {
        Self { min_scene_len,
             threshold}
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameConfig {
    output_dir: PathBuf,
    video_path: PathBuf,

    save_image: ImageFormat,
    save_frames:bool,
    save_metadata:bool,

    resize_width: bool,
    frame_size:(u32,u32),

    scene_detection: HashMap<String, SceneDetectionConfig>,

    quality_threshold:f32,
    similarity_threshold:f32,

    parallel_extract:bool,

    categories: HashMap<String, Vec<String>>,
}

impl Default for FrameConfig {
    fn default() -> Self {
        let mut scene_detection = HashMap::new();
        scene_detection.insert(
            "fast".to_string(),
            SceneDetectionConfig::new(30, 0.3)
        );
        scene_detection.insert(
            "normal".to_string(),
            SceneDetectionConfig::new(27, 0.5)
        );
        
        scene_detection.insert(
            "slow".to_string(),
            SceneDetectionConfig::new(25, 1)
        );

        let mut categories = HashMap::new();
        categories.insert(
            "key_frames".to_string(),
            vec!["motion".to_string(), "object".to_string(), "quality".to_string()]
        );
        categories.insert(
            "scene_changes".to_string(),
            vec!["content".to_string()]
        );
        categories.insert(
            "quality_frames".to_string(),
            vec!["quality".to_string()]
        );
        categories.insert(
            "regular_frames".to_string(),
            vec!["motion".to_string(), "object".to_string()]
        );

        Self {
            output_dir: PathBuf::from("./assets/keyframes"),
            video_path: PathBuf::from("./assets/video"),
            save_image: "jpg".to_string(),
            save_frames: true,
            save_metadata: true,
            resize_width: true,
            frame_size: (640, 480),
            scene_detection,
            quality_threshold: 0.6,
            similarity_threshold: 0.85,
            parallel_extract: true,
            categories,
        }
    }
}

impl FrameConfig {
    pub fn new()->Self{
        Self::default()
    }

    fn validate_config(&self)->Result<()>{
        if !(0.0..=1.0).contains(&self.quality_threshold){
            return Err(anyhow::anyhow!("quality_threshold must be between 0 and 1"));
        }
        if !(0.0..=1.0).contains(&self.similarity_threshold){
            return Err(anyhow::anyhow!("similarity_threshold must be between 0 and 1"));
        }

        for (speed, config) in &self.scene_detection {
            if !(0.0..=100.0).contains(&config.threshold) {
                return Err(anyhow::anyhow!("threshold must be between 0 and 100 for {}", speed));
            }
            if !(0.0..=10.0).contains(&config.min_scene_len) {
                return Err(anyhow::anyhow!("min_scene_len must be between 0 and 10 for {}", speed));
            }   
        }
        self.validate_frame_size()?;
        self.validate_categories()?;
        Ok(())
    }

    pub fn load_from_file<P:AsRef<Path>>(path:P)->Result<Self>{
        let config_path = path.as_ref();
        if !config_path.exists(){
            warn!("config file not found {:?}, using default config", config_path);
            return Ok(Self::default());
        }
        let config_str = std::fs::read_to_string(config_path)
            .context(format!("failed to read config file {:?}", config_path))?; 

        let config:Self = serde_json::from_str(&config_str)
            .context(format!("failed to parse config file {:?}", config_str))?;

        config.validate_config()?;
        Ok(config)
    }

    pub fn save_to_file<P:AsRef<Path>>(&self, path:P)->Result<()>{
        let config_path = path.as_ref();
        if let Some(parent) = config_path.parent(){
            std::fs::create_dir_all(parent)
                .context(format!("failed to create parent directory {:?}", parent))?;
        }
        let config_str = serde_json::to_string_pretty(self)
            .context(format!("failed to serialize config to json"))?;

        std::fs::write(config_path, config_str)
            .context(format!("failed to write config to {:?}", config_path))?;
        info!("config saved to {:?}", config_path);
        Ok(())
    }   

    pub fn update(&mut self, updates:HashMap<String, serde_json::Value>)->Result<()>{
        for (key, value) in updates {
            match key.as_str() {
                "output_dir" => {
                    if let Some(path) = value.as_str(){
                        self.output_dir = PathBuf::from(path);
                    }
                }
                "video_path" => {
                    if let Some(path) = value.as_str(){
                        self.video_path = PathBuf::from(path);
                    }
                }
                "save_image" => {
                    if let Some(format) = value.as_str(){
                        self.save_image = serde_json::from_str(&format.to_string())
                            .context(format!("invalid image format {:?}", format))?;
                    }
                }
                "save_frames" => {
                    if let Some(save_frames) = value.as_bool(){
                        self.save_frames = save_frames;
                    }
                }
                _ => warn!("unknown key {:?}", key),
            }
        }
        self.validate_config()?;
        Ok(())  
    }

    pub fn output_dir(&self)->&PathBuf{     
        &self.output_dir
    }

    pub fn video_path(&self)->&PathBuf{
        &self.video_path
    }

    fn validate_frame_size(&self)->Result<()>{
        let (width, height) = self.frame_size;
        if width == 0 || height == 0{
            return Err(anyhow::anyhow!("frame_size must be greater than 0"));
        }
        if width >7680 || height >4320{
            return Err(anyhow::anyhow!("frame_size must be less than 8K"));
        }
        Ok(())
    }

    fn validate_categories(&self)->Result<()>{
        let valid_types:Vec<&str> = vec!["motion", "object", "quality", "content"];
        for (category, types) in &self.categories{
            for type_name in types{
                if !valid_types.contains(&type_name.as_str()){
                    return Err(anyhow::anyhow!("invalid type {:?} in category {:?}", type_name, category));
                }
            }
            
        }
        let required_categories = [
            "key_frames",
            "scene_changes",
            "quality_frames",
            "regular_frames",
        ];
        for category in required_categories.iter(){
            if !self.categories.contains_key(*category){
                return Err(anyhow::anyhow!("required category {:?} not found", category));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat{
    JPG,
    PNG,
    WEBP,
}

impl Default for ImageFormat{
    fn default()->Self{
        ImageFormat::JPG
    }
}
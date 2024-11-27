from pathlib import Path
from .utils.logger_config import LoggerManager
from typing import Optional
from .bridge import PyFrameBridge

class AppInitializer:
    """应用程序初始化器"""
    _instance: Optional['AppInitializer'] = None

    def __init__(self, config_path: Optional[str] = None):
        """初始化应用程序初始化器"""
        self.logger = None
        self.frame_bridge = None
        self.config_path = config_path or "config.json"

    def init_app(self)->None:
        """初始化应用程序"""
        try:
            self.logger = LoggerManager.init_app_logger()
            self.logger.info("App initialized")

            self.frame_bridge = PyFrameBridge.get_instance(self.config_path)
            self.logger.info(f"FrameBridge initialized: {self.frame_bridge}")

            self._setup_default_config()

            self.logger.info("App initialized")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize app: {str(e)}")
            raise 

    def _setup_default_config(self)->None:
        """设置默认配置"""
        try:
            default_config = {
                "output_dir": str(Path("./output").absolute()),
                "video_path": str(Path("./video.mp4").absolute()),
                "save_frames":True
            }
            self.frame_config.update_config(default_config)
            self.logger.info(f"Default config setup: {self.frame_config}")

            self.frame_config.save_config(self.config_path)
            self.logger.info(f"Default config saved to {self.config_path}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to setup default config: {str(e)}")
            raise 
    
    def get_instance(cls, config_path: Optional[str] = None)->'AppInitializer':
        """获取应用程序初始化器实例"""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance
    
def init_app(config_path: Optional[str] = None)->None:
    """初始化应用程序"""
    try:
        app = AppInitializer.get_instance(config_path).init_app()
        return app
    except Exception as e:
        print(f"Failed to initialize app: {str(e)}")
        raise 

if __name__ == "__main__":
    try:
       app = init_app()
       print(f"App initialized: {app}")
    except Exception as e:
        print(f"Failed to initialize app: {str(e)}")
        raise 
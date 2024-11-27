from typing import Optional, Dict, Any
from pathlib import Path
from threading import Lock

# import rust_bridge module with three dots to indicate the parent directory
from ...rust_bridge import PyFrameConfig

class FrameBridge:
    """桥接Rust的FrameConfig配置"""
    _instance: Optional['FrameBridge'] = None
    _lock = Lock()

    def __new__(cls, config_path: Optional[str] = None) -> 'FrameBridge':
        """控制实例创建的方法"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化FrameConfig桥接器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        try:
            if not hasattr(self, "_initialized"):
                if config_path is None:
                    self.config = PyFrameConfig()
                else:
                    self.config = PyFrameConfig.load_from_file(config_path)
                self._initialized = True
        except Exception as e:
            raise Exception(f"Failed to initialize FrameConfig: {str(e)}") from e

    def update_config(self, update: Dict[str, Any])->None:
        """更新配置
        
        Args:
            update: 更新内容
        Raises:
            Exception: 更新配置失败时抛出异常
        """
        try:
            self.config.update(update)
        except Exception as e:
            raise Exception(f"Failed to update FrameConfig: {str(e)}") from e

    def save_config(self, path: str)->None:
        """保存配置到文件
        
        Args:
            path: 保存配置的文件路径
            
        Raises:
            Exception: 保存配置失败时抛出异常
        """
        try:
            self.config.save_to_file(path)
        except Exception as e:
            raise Exception(f"Failed to save FrameConfig: {str(e)}") from e
        
    @property
    def video_path(self)->str:
        """获取视频路径"""
        return self.config.video_path

    @property
    def output_dir(self)->str:
        """获取输出目录"""
        return self.config.output_dir
    
    @classmethod
    def get_instance(cls, config_path: Optional[str] = None)->'FrameBridge':
        """获取 FrameConfigBridge 实例
        
        Args:
            config_path: 可选的配置文件路径
            
        Returns:
            FrameConfigBridge 实例
        """
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance
    
    def __str__(self)->str:
        """获取配置的字符串表示"""
        return f"FrameConfigBridge(video_path={self.video_path}, output_dir={self.output_dir})"
    
    def __repr__(self)->str:
        """获取配置的详细字符串表示"""
        return self.__str__()
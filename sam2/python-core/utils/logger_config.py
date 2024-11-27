from enum import Enum
from ..bridge import LoggerBridge

class LoggerMode(Enum):
    ContentExtractor = "content_extractor"
    

class LoggerManager:
    """日志管理器"""
    _initialized = False

    @classmethod
    def init_app_logger(cls):
        """应用程序启动时初始化全局logger配置"""
        if not cls._initialized:
            LoggerBridge(
                log_dir="logs",
                log_level="INFO",
                max_size=10,
                max_files=5
            )
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, mode: LoggerMode)->LoggerBridge:
        """根据模式获取对应的logger
        
        Args:
            mode: 日志模式
            
        Returns:
            LoggerBridge: logger实例
        """
        if not cls._initialized:
            cls.init_app_logger()
        
        # 不同模式下的日志配置
        log_configs = {
            LoggerMode.ContentExtractor: {
                "log_dir": "logs/content_extractor",
                "log_level": "INFO",
                "max_size": 10,
                "max_files": 5
            }
        }

        config = log_configs.get(mode)
        return LoggerBridge.get_logger(log_dir=config["log_dir"],
                                       log_level=config["log_level"],
                                       max_size=config["max_size"],
                                       max_files=config["max_files"])
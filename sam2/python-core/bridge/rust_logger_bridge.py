from typing import Optional, Dict, Any
from threading import Lock

# import rust_core module with three dots to indicate the parent directory
from ...rust_bridge import LoggerConfig, Logger

# 桥接Rust的Logger功能
class LoggerBridge:
    """桥接Rust的Logger功能"""
    _instance: Optional['LoggerBridge'] = None
    _lock = Lock()

    # 控制实例创建的方法
    def __new__(cls, log_dir: str= "logs") -> 'LoggerBridge':
        """控制实例创建的方法"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._log_dir = log_dir
        return cls._instance
    
    # 初始化Logger桥接器
    def __init__(self, log_level: str = "INFO",
                 max_size: int = 10,
                 max_files: int = 5):
        """初始化Logger桥接器
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
            max_size: 最大文件大小(MB)
            max_files: 保留的文件数量
        """
        try:
            if not hasattr(self,'_initialized'):
                self.config = LoggerConfig(self._log_dir)
                self.config.log_level = log_level
                self.config.max_size = max_size
                self.config.max_files = max_files
                self.logger = Logger(self.config)
                self._initialized = True

        except Exception as e:
            raise(f"Error initializing LoggerBridge: {str(e)}")
    
    def log(self, message: str):
        """记录日志"""
        self.logger.log(message)

    def warn(self, message: str):
        """记录警告日志"""
        self.logger.warn(message)

    def error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
    
    # 获取Logger实例
    @classmethod
    def get_logger(cls, log_dir: str = "logs")->'LoggerBridge':
        """获取Logger实例
        
        Args:
            log_dir: 日志目录路径
            
        Returns:
            LoggerBridge: Logger桥接器实例
        """
        if cls._instance is None:
            cls._instance = cls(log_dir)
        return cls._instance
# services/base.py
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from threading import Lock
from ..bridge.rust_logger_bridge import LoggerBridge
from ..bridge.rust_frame_bridge import FrameBridge
from .message import MessageBus, Message, MessageType

# 服务状态的定义，用于描述服务的状态，类似一个状态机
class ServiceState:
    """服务状态"""
    INIT = "initialized" # 初始化
    RUNNING = "running" # 运行中
    STOPPED = "stopped" # 停止
    ERROR = "error" # 错误

# 服务基类，定义了服务的基本功能和状态管理
class BaseService(ABC):
    """服务基类"""
    # 初始化服务， 需要将服务名称、日志桥接器和帧配置桥接器作为参数，因为这些是服务依赖的外部资源
    def __init__(self, service_name: str, logger: LoggerBridge, frame_bridge: FrameBridge):
        """初始化服务
        
        Args:
            service_name: 服务名称
            logger: 日志桥接器
            frame_bridge: 帧配置桥接器
        """
        self.service_name = service_name
        self.logger = logger
        self.frame_bridge = frame_bridge
        self.message_bus = MessageBus() # 消息总线，用于服务之间的通信
        self.state = ServiceState.INIT # 服务状态
        self._lock = Lock() # 锁，用于线程同步
        self._error_count = 0 # 错误计数
        
        self.logger.info(f"Service {service_name} initialized")

    @abstractmethod
    async def start(self) -> None:
        """启动服务"""
        with self._lock: # 加锁，防止并发问题
            if self.state != ServiceState.INIT: # 如果服务状态不是初始化
                raise RuntimeError(f"Service {self.service_name} is already started")
            self.state = ServiceState.RUNNING # 设置服务状态为运行中
            self.logger.info(f"Service {self.service_name} started")

    @abstractmethod
    async def stop(self) -> None:
        """停止服务"""
        with self._lock: # 加锁，防止并发问题
            if self.state != ServiceState.RUNNING: # 如果服务状态不是运行中
                raise RuntimeError(f"Service {self.service_name} is not running")
            self.state = ServiceState.STOPPED
            self.logger.info(f"Service {self.service_name} stopped")

    # 发布消息，将消息发布到消息总线，并记录日志
    def publish_message(self, message: Message) -> None:
        """发布消息
        
        Args:
            message: 要发布的消息
        """
        try:
            self.message_bus.publish(message) # 发布消息
            self.logger.debug(f"Published message: {message}")
        except Exception as e:
            self._handle_error(f"Failed to publish message: {str(e)}")

    def subscribe_message(self, message_type: MessageType, callback: callable) -> None:
        """订阅消息
        
        Args:
            message_type: 消息类型
            callback: 回调函数
        """
        try:
            self.message_bus.subscribe(message_type, callback) # 订阅消息
            self.logger.debug(f"Subscribed to message type: {message_type}")
        except Exception as e:
            self._handle_error(f"Failed to subscribe message: {str(e)}")

    def _handle_error(self, error_msg: str) -> None:
        """处理错误
        
        Args:
            error_msg: 错误信息
        """
        self._error_count += 1 # 错误次数加1
        self.logger.error(f"Service {self.service_name}: {error_msg}") # 记录错误日志
        if self._error_count >= 3:  # 错误次数超过阈值
            self.state = ServiceState.ERROR # 设置服务状态为错误    
            self.logger.error(f"Service {self.service_name} entered error state")

    # 检查服务是否运行中
    @property
    def is_running(self) -> bool:
        """检查服务是否运行中"""
        return self.state == ServiceState.RUNNING

    # 检查服务是否健康
    @property
    def is_healthy(self) -> bool:
        """检查服务是否健康"""
        return self.state in [ServiceState.INIT, ServiceState.RUNNING]

    # 获取服务状态信息
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态信息"""
        return {
            "service_name": self.service_name,
            "state": self.state,
            "error_count": self._error_count,
            "is_healthy": self.is_healthy
        }

    # 返回服务的字符串表示
    def __str__(self) -> str:
        """返回服务的字符串表示"""
        return f"{self.service_name}(state={self.state})"
# services/message.py
from dataclasses import dataclass, field
from enum import Enum, auto
import traceback
from typing import Any, Dict, Optional, TypeVar,Generic, Literal, TypedDict, Union
from datetime import datetime
import uuid

T = TypeVar('T')

class ToolStatus(Enum):
    """工具状态"""
    INIT = 'init'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    STOPPED = 'stopped'
    ERROR = 'error'

class SystemStatus(Enum):
    """系统状态"""
    ALL_RUNNING = 'all_running'
    ALL_COMPLETED = 'all_completed'
    ALL_FAILED = 'all_failed'
    ALL_STOPPED = 'all_stopped'
    ALL_ERROR = 'all_error'
    # PARTIAL_COMPLETED = 'partial_completed'
    # PARTIAL_FAILED = 'partial_failed'
    # PARTIAL_STOPPED = 'partial_stopped'
    # PARTIAL_ERROR = 'partial_error'
    MIXED_STATUS = 'mixed_status'
    PARTIAL_RUNNING = 'partial_running'
    

# 消息优先级的定义，用于描述消息的优先级,枚举类型
class MessagePriority(Enum):
    """消息优先级"""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()

# 将系统消息跟单个业务消息分开定义， 综合来看， 系统消息的优先级最高
class MessageType(Enum):
    """消息类型"""
    # 服务生命周期消息
    SERVICE_STARTED = "service_started" # 服务启动
    SERVICE_STOPPED = "service_stopped" # 服务停止
    SERVICE_ERROR = "service_error" # 服务错误
    
    # 帧提取相关消息
    FRAME_EXTRACTION_STARTED = "frame_extraction_started" # 帧提取开始
    FRAME_EXTRACTED = "frame_extracted" # 帧提取完成
    FRAME_EXTRACTION_FAILED = "frame_extraction_failed" # 帧提取失败
    
    # 帧处理相关消息
    FRAME_PROCESSING_STARTED = "frame_processing_started" # 帧处理开始
    FRAME_PROCESSED = "frame_processed" # 帧处理完成
    FRAME_PROCESSING_FAILED = "frame_processing_failed" # 帧处理失败
    
    TOOL_STATUS = "tool_status" # 工具状态

    SYSTEM_STATUS = "system_status" # 系统状态
    
    # Rust交互相关消息
    RUST_DATA_SEND = "rust_data_send" # Rust发送数据
    RUST_DATA_RECEIVED = "rust_data_received" # Rust接收数据
    RUST_ERROR = "rust_error"

tool_status_literals = Literal["init", "running", "completed", "failed", "stopped", "error"]
system_status_literals = Literal["all_running", "all_completed", "all_failed", "all_stopped", "all_error", "mixed_status", "partial_running"]
    
class MessageMetadata(TypedDict, total=False):
    """消息元数据类型定义
    
    Attributes:
        tool_id: 工具ID(可选)
        status: 状态值
        timestamp: 时间戳
        video_path: 视频路径（可选）
        frame_count: 帧数（可选）
        error_type: 错误类型（可选）
        error_message: 错误信息（可选）
        context: 上下文信息（可选）
        tool_statuses: 工具状态字典（可选，用于系统状态消息）
    """
    tool_id:Optional[str]
    status: Union[tool_status_literals, system_status_literals]
    timestamp:datetime
    video_path:Optional[str]
    frame_count:Optional[int]
    error_type:Optional[str]
    error_message:Optional[str]
    context:Optional[Dict[str, Any]]
    tool_statuses:Optional[Dict[str, str]]

@dataclass
class Message(Generic[T]):
    """消息基类
    
    Attributes:
        id: 消息唯一标识
        type: 消息类型
        data: 消息数据
        source: 消息来源服务
        timestamp: 消息时间戳
        priority: 消息优先级
        metadata: 元数据字典
        correlation_id: 关联消息ID,用于追踪相关消息
    
    Example:
        >>> msg = Message(
        ...     type=MessageType.TOOL_STATUS,
        ...     data={"frames": [...]},
        ...     source="frame_extractor"
        ... )
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))                    # 消息唯一标识
    type: MessageType          # 消息类型
    data: T                    # 消息数据
    source: str                # 消息来源服务
    timestamp: datetime = field(default_factory=datetime.now)        # 消息时间戳
    priority: MessagePriority = field(default=MessagePriority.NORMAL)  # 消息优先级
    metadata: MessageMetadata = field(default_factory=dict)  # 元数据
    correlation_id: Optional[str] = field(default=None)  # 关联消息ID
    
    

    def add_metadata(self, key: str, value: Any) -> None:
        """添加元数据"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)

    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"Message(id={self.id}, "
            f"type={self.type.value}, "
            f"source={self.source}, "
            f"priority={self.priority.name}, "
            f"timestamp={self.timestamp})"
        )

# 特定消息类型的工厂函数
def create_tool_status_message(
    tool_id: str,
    status: ToolStatus,
    frames: Any = None,
    source: str = "",
    video_path: str = "",
    frame_count: int = 0,
    priority: MessagePriority = MessagePriority.NORMAL,
    **additional_metadata: Dict[str, Any]
) -> Message[Any]:
    """创建工具状态消息
    
    Args:
        tool_id: 工具ID
        status: 工具状态
        frames: 帧数据（可选）
        source: 消息来源
        video_path: 视频路径（可选）
        frame_count: 帧数（可选）
        priority: 消息优先级
        additional_metadata: 额外的元数据
    """
    metadata = {
        "tool_id": tool_id,
        "status": status.value,
        "timestamp": datetime.now(),
        "video_path": video_path,
        "frame_count": frame_count,
        **additional_metadata
    }
    return Message(
        type=MessageType.TOOL_STATUS,
        data=frames,
        source=source,
        priority=priority,
        metadata=metadata
    )


# 系统状态消息类型的工厂函数
def create_system_status_message(
    status: SystemStatus,
    tool_status: Dict[str, ToolStatus],
    source: str = "",
    priority: MessagePriority = MessagePriority.NORMAL,
    **additional_metadata: Dict[str, Any]
) -> Message[None]:
    """创建工具状态消息
    
    Args:
        tool_id: 工具ID
        status: 工具状态
        frames: 帧数据（可选）
        source: 消息来源
        priority: 消息优先级
        additional_metadata: 额外的元数据
    """
    metadata = {
        "status": status.value,
        "tool_statuses":{tid:st.value for tid, st in tool_status.items()},
        "timestamp": datetime.now(),
        **additional_metadata
    }
    return Message(
        type=MessageType.SYSTEM_STATUS,
        data=None,
        source=source,
        priority=priority,
        metadata=metadata
    )


def create_error_message(
    error: Exception,
    source: str,
    context: Dict[str, Any],
    correlation_id: Optional[str] = None,
    tool_id: Optional[str] = None
) -> Message[None]:
    """创建错误消息"""
    metadata = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "context": context
    }
    if tool_id:
        metadata["tool_id"] = tool_id

    return Message(
        type=MessageType.SERVICE_ERROR,
        data=None,
        source=source,
        priority=MessagePriority.HIGH,
        metadata=metadata,
        correlation_id=correlation_id
    )
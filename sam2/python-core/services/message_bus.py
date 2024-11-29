# services/message_bus.py
from typing import Dict, List, Callable, Optional
from threading import Lock
from queue import Queue
import asyncio
from datetime import datetime
import time

from .message import Message, MessageType, MessagePriority
from ..bridge.rust_logger_bridge import LoggerBridge

class MessageBus:
    """消息总线实现"""
    _instance = None
    _lock = Lock()

    def __new__(cls) -> 'MessageBus':
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """初始化消息总线"""
        if not self._initialized:
            # 订阅者字典，按消息类型存储回调函数， 用于消息分发
            self._subscribers: Dict[MessageType, List[Callable]] = {}
            # 消息队列，按优先级分类， 用于消息缓存
            self._queues: Dict[MessagePriority, Queue] = {
                MessagePriority.HIGH: Queue(),
                MessagePriority.NORMAL: Queue(),
                MessagePriority.LOW: Queue()
            }
            # 消息历史记录
            self._message_history: List[Message] = []
            # 最大历史记录数
            self._max_history_size = 1000
            # 获取日志实例
            self.logger = LoggerBridge.get_instance()
            # 初始化完成标记
            self._initialized = True
            # 运行标志
            self._running = False
            # 创建事件循环
            self._loop = asyncio.get_event_loop()
            # 最大并发处理消息数
            self._max_concurrent_messages = 100
            self._semaphore = asyncio.Semaphore(self._max_concurrent_messages)
            self._metrics = {
                'processed': 0,
                'failed': 0,
                'processing_time': []
            }

    async def start(self):
        """启动消息总线"""
        self._running = True
        self.logger.info("MessageBus started")
        # 启动消息处理循环
        await self._process_messages()

    async def stop(self):
        """停止消息总线"""
        self._running = False
        self.logger.info("MessageBus stopped")

    # 发布消息
    def publish(self, message: Message) -> None:
        """发布消息
        
        Args:
            message: 要发布的消息
        """
        try:
            # 添加到相应优先级的队列
            self._queues[message.priority].put(message)
            # 记录消息历史， 用于消息回溯，表示已经见过的消息
            self._record_message(message)
            self.logger.debug(f"Published message: {message}")
            
            # 异步处理消息， 使用create_task创建任务， 避免阻塞当前线程
            asyncio.create_task(self._handle_message(message))
            
        except Exception as e:
            self.logger.error(f"Error publishing message: {str(e)}")
            raise
    
    # 主要是订阅消息
    def subscribe(self, message_type: MessageType, callback: Callable[[Message], None]) -> None:
        """订阅消息
        
        Args:
            message_type: 消息类型
            callback: 回调函数
        """
        try:
            # 如果消息类型不存在， 则创建一个空列表
            if message_type not in self._subscribers:
                self._subscribers[message_type] = []
            # 将回调函数添加到消息类型的订阅者列表中
            self._subscribers[message_type].append(callback)
            self.logger.debug(f"Subscribed to {message_type.value}")
        except Exception as e:
            self.logger.error(f"Error subscribing to message: {str(e)}")
            raise

    def unsubscribe(self, message_type: MessageType, callback: Callable) -> None:
        """取消订阅
        
        Args:
            message_type: 消息类型
            callback: 回调函数
        """
        try:
            if message_type in self._subscribers:
                self._subscribers[message_type].remove(callback)
                self.logger.debug(f"Unsubscribed from {message_type.value}")
        except Exception as e:
            self.logger.error(f"Error unsubscribing from message: {str(e)}")
            raise

    # 处理消息队列
    async def _process_messages(self):
        """处理消息队列"""
        while self._running:
            # 按优先级处理消息
            for priority in MessagePriority:
                queue = self._queues[priority]
                while not queue.empty():
                    message = queue.get()
                    await self._handle_message(message)
            # 避免CPU过度使用
            await asyncio.sleep(0.01)

    # 处理单个消息
    async def _handle_message(self, message: Message):
        """处理单个消息"""
        async with self._semaphore:
            start_time = time.time()
            try:
                if message.type in self._subscribers:
                    for callback in self._subscribers[message.type]:
                        try:
                            # 异步执行回调
                            if asyncio.iscoroutinefunction(callback):
                                await callback(message)
                            else:
                                callback(message)
                            self._metrics['processed'] += 1
                        except asyncio.TimeoutError:
                            self.logger.error(
                                f"Subscriber callback timed out: {callback}, "
                                f"message: {message}"
                            )
                            self._metrics['failed'] += 1
                        except Exception as e:
                            self.logger.error(
                                f"Error in subscriber callback: {str(e)}, "
                                f"message: {message}"
                            )
                            self._metrics['failed'] += 1
                            
            finally:
                end_time = time.time()
                self._metrics['processing_time'].append(end_time - start_time)
                        

    def _record_message(self, message: Message):
        """记录消息历史"""
        self._message_history.append(message)
        # 维护历史记录大小
        if len(self._message_history) > self._max_history_size:
            # 如果历史记录超过最大值， 则删除最早的消息
            self._message_history.pop(0)

    def get_message_history(self, 
                          message_type: Optional[MessageType] = None, 
                          start_time: Optional[datetime] = None) -> List[Message]:
        """获取消息历史
        
        Args:
            message_type: 可选的消息类型过滤
            start_time: 可选的开始时间过滤
            
        Returns:
            List[Message]: 过滤后的消息历史
        """
        messages = self._message_history
        
        if message_type:
            messages = [m for m in messages if m.type == message_type]
            
        if start_time:
            messages = [m for m in messages if m.timestamp >= start_time]
            
        return messages

    def clear_history(self):
        """清除消息历史"""
        self._message_history.clear()
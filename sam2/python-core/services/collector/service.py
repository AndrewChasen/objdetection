# services/collector/service.py
from typing import List, Dict, Any
import asyncio
from datetime import datetime
import uuid
from ..base import BaseService
from ..message import Message
from ...bridge import LoggerBridge, FrameBridge

class CollectorService(BaseService):
    def __init__(self, service_name: str, logger: LoggerBridge, frame_bridge: FrameBridge):
        super().__init__(service_name, logger, frame_bridge)
        self._batch_size = 100  # 批次大小
        self._batch_timeout = 30  # 批次超时时间(秒)
        self._current_batch: List[Any] = [] # 当前批次
        self._batch_lock = asyncio.Lock() # 批次锁
        self._last_batch_time = datetime.now() # 上次批次时间
        
    async def _handle_processed_frames(self, message: Message):
        """处理帧批次"""
        async with self._batch_lock:
            frames = message.data
            self._current_batch.extend(frames)
            
            # 检查是否需要发送批次
            if self._should_send_batch():
                await self._send_batch()
    
    def _should_send_batch(self) -> bool:
        """检查是否应该发送批次"""
        # 批次大小达到阈值
        if len(self._current_batch) >= self._batch_size:
            return True
            
        # 批次超时
        time_elapsed = (datetime.now() - self._last_batch_time).seconds
        if time_elapsed >= self._batch_timeout and self._current_batch:
            return True
            
        return False
    
    async def _send_batch(self):
        """发送批次到Rust"""
        if not self._current_batch:
            return
            
        try:
            batch_data = {
                "frames": self._current_batch,
                "timestamp": datetime.now(),
                "batch_id": str(uuid.uuid4())
            }
            
            # 发送到Rust
            await self.frame_bridge.send_batch(batch_data)
            
            # 清理已发送的批次
            self._current_batch = []
            self._last_batch_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to send batch: {str(e)}")
            raise
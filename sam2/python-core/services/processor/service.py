# services/processor/service.py
from typing import Dict, List, Any
import asyncio
from pathlib import Path

from ..base import BaseService
from ..message import Message, MessageType, MessagePriority
from ...bridge import LoggerBridge, FrameBridge

class ProcessorService(BaseService):
    """处理器服务：处理从提取器接收到的帧"""
    
    def __init__(self, service_name: str, logger: LoggerBridge, frame_bridge: FrameBridge):
        super().__init__(service_name, logger, frame_bridge)
        self._active_tasks: Dict[str, asyncio.Task] = {}
        
        # 订阅提取器的消息
        self.subscribe_message(
            MessageType.FRAME_EXTRACTED,
            self._handle_extracted_frames
        )

    async def _handle_extracted_frames(self, message: Message):
        """处理提取到的帧"""
        frames = message.data
        video_path = message.metadata["video_path"]
        task_id = f"process_{Path(video_path).stem}"
        
        try:
            # 创建处理任务
            task = asyncio.create_task(
                self._process_frames_task(task_id, frames, video_path)
            )
            # 将任务添加到活动任务列表中
            self._active_tasks[task_id] = task
            await task
            
        except Exception as e:
            self.logger.error(f"Processing failed for {video_path}: {str(e)}")
            raise

    async def _process_frames_task(self,
                                 task_id: str,
                                 frames: List[Any],
                                 video_path: str) -> None:
        """处理帧任务"""
        try:
            # 发布处理开始消息
            self.publish_message(Message(
                type=MessageType.FRAME_PROCESSING_STARTED,
                data={"frame_count": len(frames)},
                source=self.service_name,
                metadata={"task_id": task_id, "video_path": video_path}
            ))
            
            # 处理帧
            processed_frames = await self._process_frames(frames)
            
            # 发布处理完成消息
            self.publish_message(Message(
                type=MessageType.FRAME_PROCESSED,
                data=processed_frames,
                source=self.service_name,
                priority=MessagePriority.HIGH,
                metadata={
                    "task_id": task_id,
                    "video_path": video_path,
                    "frame_count": len(processed_frames)
                }
            ))
            
        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            raise
        finally:
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]

    async def _process_frames(self, frames: List[Any]) -> List[Any]:
        """实际的帧处理逻辑"""
        # 这里可以添加具体的处理逻辑
        return frames
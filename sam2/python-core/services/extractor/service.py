# services/extractor/service.py
from typing import List, Dict
import asyncio
from pathlib import Path

from ..base import BaseService
from ..message import Message, MessageType, MessagePriority, create_frame_extracted_message, create_error_message
from ...bridge import LoggerBridge, FrameBridge
from .extractors.base import BaseExtractor
from .extractors.content import ContentDetectorExtractor

class ExtractorService(BaseService):
    """提取器服务实现"""
    
    def __init__(self, service_name: str, logger: LoggerBridge, frame_bridge: FrameBridge):
        """初始化提取器服务
        
        Args:
            service_name: 服务名称
            logger: 日志桥接器
            frame_bridge: 帧桥接器
        """
        super().__init__(service_name, logger, frame_bridge)
        self._extractors: Dict[str, BaseExtractor] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._initialize_extractors()

    def _initialize_extractors(self):
        """初始化所有提取器"""
        try:
            # 注册内容检测提取器
            self.register_extractor(
                "content",
                ContentDetectorExtractor()
            )
            self.logger.info("Extractors initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize extractors: {str(e)}")
            raise
    
    # 注册提取器,本质上就是添加到对应的字典上去，
    def register_extractor(self, name: str, extractor: BaseExtractor):
        """注册提取器
        
        Args:
            name: 提取器名称
            extractor: 提取器实例
        """
        self._extractors[name] = extractor
        self.logger.debug(f"Registered extractor: {name}")

    async def start(self) -> None:
        """启动提取器服务"""
        await super().start()
        # 可以在这里添加额外的启动逻辑
        self.logger.info(f"ExtractorService {self.service_name} started")

    async def stop(self) -> None:
        """停止提取器服务"""
        # 取消所有正在运行的任务
        for task_id, task in self._active_tasks.items():
            if not task.done():
                task.cancel()
                self.logger.info(f"Cancelled extraction task: {task_id}")
        
        await super().stop()
        self.logger.info(f"ExtractorService {self.service_name} stopped")

    async def extract_frames(self, 
                           video_path: str, 
                           extractor_name: str = "content",
                           **kwargs) -> None:
        """提取视频帧
        
        Args:
            video_path: 视频文件路径
            extractor_name: 使用的提取器名称
            **kwargs: 额外的提取参数
        """
        task_id = f"extract_{Path(video_path).stem}_{extractor_name}"
        
        try:
            # 检查视频文件
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # 检查提取器
            if extractor_name not in self._extractors:
                raise ValueError(f"Extractor not found: {extractor_name}")

            # 发布任务开始消息
            self.publish_message(Message(
                type=MessageType.FRAME_EXTRACTION_STARTED,
                data={"video_path": video_path},
                source=self.service_name,
                metadata={"task_id": task_id}
            ))

            # 创建提取任务
            task = asyncio.create_task(
                self._extract_frames_task(
                    task_id, video_path, extractor_name, **kwargs
                )
            )
            self._active_tasks[task_id] = task
            
            # 等待任务完成
            await task

        except Exception as e:
            self.logger.error(f"Extraction failed for {video_path}: {str(e)}")
            # 发布错误消息
            self.publish_message(create_error_message(
                error=e,
                source=self.service_name,
                context={
                    "video_path": video_path,
                    "extractor": extractor_name,
                    "task_id": task_id
                }
            ))
            raise

    async def _extract_frames_task(self,
                                 task_id: str,
                                 video_path: str,
                                 extractor_name: str,
                                 **kwargs) -> None:
        """执行帧提取任务
        
        Args:
            task_id: 任务ID
            video_path: 视频文件路径
            extractor_name: 提取器名称
            **kwargs: 额外参数
        """
        try:
            extractor = self._extractors[extractor_name]
            frames = await extractor.extract(video_path, **kwargs)
            
            # 发布提取完成消息
            self.publish_message(create_frame_extracted_message(
                frames=frames,
                source=self.service_name,
                video_path=video_path,
                frame_count=len(frames),
                priority=MessagePriority.HIGH
            ))
            
            self.logger.info(
                f"Successfully extracted {len(frames)} frames "
                f"from {video_path} using {extractor_name}"
            )
            
        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            raise
        finally:
            # 清理任务记录
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]

    def get_active_tasks(self) -> Dict[str, str]:
        """获取当前活动的任务列表
        
        Returns:
            Dict[str, str]: 任务ID到状态的映射
        """
        return {
            task_id: "running" if not task.done() else "completed"
            for task_id, task in self._active_tasks.items()
        }
    
    # 获取可用的提取器列表
    def get_available_extractors(self) -> List[str]:
        """获取可用的提取器列表
        
        Returns:
            List[str]: 提取器名称列表
        """
        return list(self._extractors.keys())
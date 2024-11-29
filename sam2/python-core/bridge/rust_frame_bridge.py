import json
from typing import Optional, Dict, Any
from pathlib import Path
from threading import Lock
import uuid
from multiprocessing import shared_memory

import numpy as np

# import rust_bridge module with three dots to indicate the parent directory
from ...rust_bridge import PyFrameConfig

# 桥接Rust的FrameConfig配置
class FrameBridge:
    """桥接Rust的FrameConfig配置"""
    _instance: Optional['FrameBridge'] = None
    _lock = Lock()

    # 控制实例创建的方法
    def __new__(cls, config_path: Optional[str] = None) -> 'FrameBridge':
        """控制实例创建的方法"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._config_path = config_path
        return cls._instance
    
    # 初始化FrameConfig桥接器
    def __init__(self):
        """初始化FrameConfig桥接器
        
        Args:
            config_path: 配置文件路径,如果为None则使用默认配置
        """
        try:
            if not hasattr(self, "_initialized"):
                if self._config_path is None:
                    self.config = PyFrameConfig()
                else:
                    self.config = PyFrameConfig.load_from_file(self._config_path)
                self._initialized = True
        except Exception as e:
            raise Exception(f"Failed to initialize FrameConfig: {str(e)}") from e

    # 更新配置
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

    # 保存配置到文件
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
        
    # 获取视频路径
    @property
    def video_path(self)->str:
        """获取视频路径"""
        return self.config.video_path

    # 获取输出目录
    @property
    def output_dir(self)->str:
        """获取输出目录"""
        return self.config.output_dir
    
    # 获取FrameConfig桥接器实例
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
    
    # 获取配置的字符串表示
    def __str__(self)->str:
        """获取配置的字符串表示"""
        return f"FrameConfigBridge(video_path={self.video_path}, output_dir={self.output_dir})"
    
    # 获取配置的详细字符串表示
    def __repr__(self)->str:
        """获取配置的详细字符串表示"""
        return self.__str__()
    
    async def send_batch(self, batch_data: Dict[str, Any])->None:
        """
        使用共享内存发送批次帧数据到Rust端
        
        Args:
            batch_data: 包含以下字段的字典:
                - frames: List[np.ndarray]  # 图像帧列表
                - timestamp: datetime
                - batch_id: str
        """
        shared_memory_blocks = []
        
        try:
            frames = batch_data["frames"]
            frame_metadata = []
            # 创建共享内存并复制数据
            for idx, frame in enumerate(frames):
                if not isinstance(frame, np.ndarray):
                    raise ValueError(f"Frame at index {idx} is not a numpy array")
                # 创建共享内存块，唯一的共存内存名称
                shm_name = f"frame_{batch_data['batch_id']}_{idx}_{uuid.uuid4().hex[:8]}"
                # 创建共享内存块，并复制数据
                shm = shared_memory.SharedMemory(
                    name=shm_name,
                    size=frame.nbytes,
                    create=True
                )
                # 将帧数据复制到共享内存块，大小和类型与帧数据相同
                frame_array = np.ndarray(
                    frame.shape,
                    dtype=frame.dtype,
                    buffer=shm.buf
                )
                # 将帧数据复制到共享内存块
                frame_array[:] = frame[:]
                # 将共享内存块添加到列表
                shared_memory_blocks.append(shm)
                # 将帧元数据添加到列表
                frame_metadata.append(
                    {
                        'shm_name':shm_name,
                        'shape':frame.shape,
                        'dtype':str(frame.dtype),
                        'nbytes':frame.nbytes,
                    }
                )
            metadata = {
                'frame_count':len(frames),
                'frame_metadata':frame_metadata,
                'timestamp':batch_data['timestamp'].isoformat(),
                'batch_id':batch_data['batch_id'],
            }
            try:
                # 发送批次到Rust,调用Rust的process_batch_shared_memory方法
                await self.config.process_batch_shared_memory(
                    metadata=json.dumps(metadata)
                )
            except Exception as e:
                raise RuntimeError(f"Failed to send batch to Rust: {str(e)}") from e
            finally:
                for shm in shared_memory_blocks:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception as e:
                        self.logger.error(f"Failed to close or unlink shared memory block: {str(e)}")   
        except Exception as e:
            raise RuntimeError(f"Failed to send batch to Rust: {str(e)}") from e
        
    async def _cleanup_shared_memory(self,shm_name: str)->None:
        """清理共享内存"""
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except Exception as e:
            self.logger.error(f"Failed to close or unlink shared memory block: {str(e)}")   

            
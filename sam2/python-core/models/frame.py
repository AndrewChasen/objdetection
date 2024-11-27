from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict
from ..bridge import LoggerBridge
import cv2
import numpy as np
# todo！！！可以进一步的优化    
class FrameError(Exception):
    """帧处理相关的自定义异常"""
    pass

@dataclass
class Frame:
    """视频帧类
    
    用于存储和管理单个视频帧的数据和元信息。
    
    Attributes:
        image (np.ndarray): 帧图像数据
        timestamp (float): 帧时间戳（秒）
        frame_index (int): 帧索引
        metadata (Dict[str, Any]): 帧元数据
        quality_score (float): 帧质量分数 (0-1)
        creation_time (datetime): 帧创建时间
        
    Examples:
        >>> frame = Frame(image=img_array, timestamp=1.5, frame_index=0)
        >>> frame.add_metadata('detection_score', 0.95)
        >>> frame.save_with_config(config, 'frame_001')
    """
    
    image: np.ndarray
    frame_index: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    creation_time: datetime = field(default_factory=datetime.now)
    logger: LoggerBridge = field(default_factory=LoggerBridge.get_logger)

    def __post_init__(self):
        """初始化后的验证"""
        self._validate_frame()
    
    def _validate_frame(self)->None:
        """验证帧的有效性"""
        try:
            if not isinstance(self.image, np.ndarray):
                raise FrameError(f"帧图像数据必须是np.ndarray类型: {type(self.image)}")
            if self.image.size == 0:
                raise FrameError("帧图像数据不能为空")
            if not 0 <= self.quality_score <= 1:
                raise FrameError(f"帧质量分数必须在0到1之间: {self.quality_score}")
            if self.timestamp < 0:
                raise FrameError(f"帧时间戳不能为负数: {self.timestamp}")
            if self.frame_index < 0:
                raise FrameError(f"帧索引不能为负数: {self.frame_index}")
            
        except Exception as e:
            self.logger.error(f"帧数据验证失败: {e}")
            raise FrameError(f"帧数据验证失败: {e}")
    
    @property
    def shape(self)->tuple:
        """获取帧图像的形状"""
        return self.image.shape
    
    @property
    def width(self)->int:
        """获取帧图像的宽度"""
        return self.shape[1]
    
    @property
    def height(self)->int:
        """获取帧图像的高度"""
        return self.shape[0]
    
    def add_metadata(self, key:str, value:Any)->None:
        """添加元数据
        
        Args:
            key: 元数据键
            value: 元数据值
        """
        self.metadata[key] = value

    def get_metadata(self, key:str,default:Any=None)->Any:
        """获取元数据
        
        Args:
            key: 元数据键
            default: 默认值
            
        Returns:
            元数据值
        """
        return self.metadata.get(key,default)
    
    def update_quality_score(self, score:float)->None:
        """更新帧质量分数
        
        Args:
            score: 新的质量分数
        Raises:
            FrameError: 当分数无效时抛出
        """
        if not 0 <= score <= 1:
            raise FrameError(f"帧质量分数必须在0到1之间: {score}")
        self.quality_score = score
    
    def save_frame(self,config:'FrameConfig', frame_name:str, category: str, source_video: str=None) -> bool:
        """使用配置保存帧
    
        Args:
            config: 帧配置对象
            frame_name: 帧文件名（不包含扩展名）
            category: 关键帧提取类别 (如 'motion', 'object', 'quality' 等)
            source_video: 源视频文件路径
            
            Returns:
                bool: 保存是否成功
                
            Examples:
            >>> frame.save_with_config(
            ...     config=config,
            ...     frame_name='frame_001',
            ...     category='motion',
            ...     source_video='videos/test.mp4'
            ... )
        """
        try:
            if self.image is None or self.image.size == 0 or not self.image.any():
                self.logger.error("帧图像数据为空，无法保存")
                return False
            
            save_dir = config.output_dir

            if source_video:
                source_video = Path(source_video)
                save_dir = save_dir / source_video.stem

            if category:
                save_dir = save_dir / category

            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 构建文件名
            timestamp = f"_time_{self.timestamp:.2f}s" if self.timestamp is not None else ""
            filename = f"{frame_name}_{self.frame_index:04d}{timestamp}"
            
            # 保存图片
            save_path = save_dir / filename
            image_path = save_path.with_suffix(f".{config.save_image_format.lower()}")

            if not isinstance(self.image, np.ndarray):
                self.logger.error(f"帧图像数据类型错误: {type(self.image)}")
                return False
            
            success = cv2.imwrite(str(image_path), self.image)

            if not success:
                self.logger.error(f"保存帧图像失败: {image_path}")
                return False
            
            if config.save_metadata:
                metadata = {
                    'frame_index': self.frame_index,
                    'timestamp': self.timestamp,
                    'quality_score': self.quality_score,
                    'creation_time': self.creation_time.isoformat(),
                    'category': category or 'unknown',
                    'extractor':self.get_metadata('extractor', 'unknown'),
                    'extract_method':self.get_metadata('extract_method', 'unknown'),
                    'extract_time':self.creation_time.isoformat(),
                    'source_video':str(source_video) if source_video else 'unknown',
                    'video_name':source_video.stem if source_video else 'unknown',
                    'image_path':str(image_path),
                    'shape':self.shape,
                    'format':config.save_image_format,
                    'additional_metadata':self.metadata
                }
                metadata_path = save_path.with_suffix('.json')
                with open(metadata_path, 'w',encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4,ensure_ascii=False)
                    
            self.logger.info(f"保存帧成功: {image_path} (类别: {category or 'unknown'})")
            return True
        except Exception as e:
            self.logger.error(f"保存帧失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
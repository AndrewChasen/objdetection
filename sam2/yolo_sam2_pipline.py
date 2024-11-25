from datetime import datetime
import json
import torch
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
from scenedetect import detect, ContentDetector,AdaptiveDetector
from scenedetect.scene_manager import save_images
from typing import List, Optional, Any, Dict, Protocol, Tuple, Type
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Union
from enum import Enum, auto
def setup_device():
    # check the device and set the default device
    device = 'cpu' if torch.backends.mps.is_available() else 'mps'
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    # print(f"Using device: {device}")
    print(f"using device:{device}")
    return torch.device(device=device)

checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg,checkpoint, device=setup_device())

def show_mask(mask, ax,obj_id=None, random_color=False):
    if random_color:
        # 这段代码生成一个随机颜色的数组，包含三个随机生成的RGB值和一个固定的alpha值0.6
        # np.random.random(3)生成一个包含三个随机浮点数的数组，这些浮点数在0到1之间
        # np.array([0.6])创建一个包含单个元素0.6的数组
        # np.concatenate将这两个数组连接起来，形成一个包含四个元素的数组
        # axis 表示
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # 这段代码生成一个颜色数组，包含三个RGB值和一个固定的alpha值0.6
        # plt.get_cmap("tab10")获取一个名为"tab10"的colormap对象
        # cmap_idx = 0 if obj_id is None else obj_id 如果obj_id为None，则设置为0，否则设置为obj_id
        # np.array([*cmap(cmap_idx)[:3], 0.6])从colormap中获取颜色，并将其与alpha值0.6组合成一个颜色数组    
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        # *cmap(cmap_idx)[:3] 解包cmap(cmap_idx)返回的RGB值，并将其与0.6组合成一个颜色数组  
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    # mask.shape[-2:] 获取mask的最后一个两个维度的大小，即高度和宽度
    h, w = mask.shape[-2:]
    # 将mask的形状调整为(h, w, 1)，并将color的形状调整为(1, 1, -1)，然后进行逐元素相乘  
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # 使用imshow函数在ax上显示mask_image，alpha=0.5表示透明度为0.5
    # 这样的目的是将mask的值与颜色进行混合，形成一个半透明的彩色mask，方便可视化    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    # 根据labels的值，将coords中的正负点分开
    # 正点为绿色，负点为红色
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    # 几种常见的切片方式：
    # pos_points[:, 0]    # 所有行，第0列å
    # pos_points[:, 1]    # 所有行，第1列
    # pos_points[0, :]    # 第0行，所有列
    # pos_points[:, :]    # 所有行，所有列（完整数组）
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    # 解包box的四个坐标值,左上角的位置坐标
    # # 边界框格式通常是：
    # box = [x0, y0, x1, y1]
    # # 其中：
    # x0, y0 = box[0], box[1]  # 左上角坐标
    # x1, y1 = box[2], box[3]  # 右下角坐标
    x0, y0 = box[0], box[1]
    # 计算box的宽度和高度
    w, h = box[2] - box[0], box[3] - box[1]
    # 打补丁，在ax上绘制一个矩形，表示边界框，类似ps中的添加一层在原来的图片上面
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', linewidth=3))

# torch.autocast("cpu", dtype=torch.bfloat16).__enter__()
# if torch.cuda.get_device_properties(0).major >= 8:
#     # 是NVIDIA专门为深度学习设计的数据格式：tf32类型开启
#     # 开启后，可以提高计算速度，但精度会降低， 在英伟达的a6000和a800上，tf32的精度与fp32相同    
#     # 在pytorch中，matmul是矩阵乘法，cudnn是cuDNN库，用于加速深度学习中的卷积神经网络（CNN）操作        
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True



# 下载视频，并获取视频信息，视频信息包括视频的宽高，帧率，总帧数等
# 视频信息是supervision库中的一个类，用于存储视频的基本信息
# 主要的作用类似于cv2.VideoCapture(0)，可以获取视频的帧率，宽高，总帧数等
# 也就是说cv2，跟supervision库中的VideoInfo类，都可以获取视频的信息，有类似的功能
# SOURCE_VIDEO = download_assets(VideoAssets.BASKETBALL)
# video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)

SCALE_FACTOR = 0.5
START_IDX = 0
END_IDX = 100


# todo: 要判断一个视频的fps，不同的视频有不同的fps，1s有多少张图片那么计算策略不一样。

BASE_DIR = Path('./assets')
VIDEO_FRAMES_DIR = BASE_DIR / 'video_frames'  # 专门存放视频抽帧的目录
SOURCE_VIDEO = BASE_DIR / 'video' / '01_dog.mp4'
OUTPUT_DIR = BASE_DIR / 'keyframes' # 存放关键帧的目录
VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)


# 获取视频的帧生成器，用于生成视频的每一帧
# 获取的每一帧我们要选择他们的关键帧，进行人工标注
# 选择关键帧的策略：
# 1. 选择视频中变化比较剧烈的帧，比如物体移动，场景变化等
# 2. 选择视频中比较有代表性的帧，比如视频中的重要场景，重要物体等
# 3. 选择视频中的关键帧，比如视频中的重要场景，重要物体等
# 选择关键帧，我们有两种方法
# 1. 通过传统方法来确定关键帧
# 2. 通过论文中的理论来训练模型来确定关键帧
# 最好的方法是：将两者结合起来

# 选择关键帧，我们使用第一种方法，通过传统方法来确定关键帧 使用
# '提取方法': {
#             '图像差异': '检测动作变化',
#             '特征提取': '检测内容变化',
#             '场景检测': '检测场景转换',
#             'CLIP分析': '检测语义变化'
#         },
# 选择关键帧，我们使用第二种方法，通过论文中的理论来训练模型来确定关键帧 todo！！！
# 选择关键帧，我们使用第三种方法，将两者结合起 todo！！！


# 首先建立一个管理类，来管理固定场景跟切换场景两个场景下的关键帧提取，将他们进行串联
# 起来，形成一个完整的流程，后续可以扩展到训练模型来确定关键帧
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """配置错误异常类"""
    pass

class ImageFormat(Enum):
    """支持的图像格式"""
    JPG = auto()
    PNG = auto()
    WEBP = auto()

@dataclass
class SceneDetectionConfig:
    """场景检测配置类"""
    threshold:float
    min_scene_len: float

#第一步，先使用固定场景，来确定关键帧
@dataclass
class FrameConfig:
    """帧处理配置类
    
    该类管理视频帧处理的所有配置参数，包括路径设置、保存选项、
    场景检测参数、阈值设���等。支持从JSON文件加载和保存配置。
    
    Attributes:
        output_dir (Path): 输出目录路径
        video_path (Path): 视频文件路径
        save_image_format (str): 保存图像的格式，默认'jpg'
        save_frames (bool): 是否保存帧图像
        save_metadata (bool): 是否保存元数据
        scene_detection (Dict[str, SceneDetectionConfig]): 场景检测配置
            - fast: 快速场景的检测参数
            - normal: 普通场景的检测参数
            - slow: 慢速场景的检测参数
        quality_threshold (float): 质量检测阈值
        similarity_threshold (float): 相似度检测阈值
        parallel_extract (bool): 是否启用并行提取
        categories (Dict[str, List[str]]): 帧分类配置
            - key_frames: 关键帧类型列表
            - scene_changes: 场景变化类型列表
            - quality_frames: 质量帧类型列表
            - regular_frames: 常规帧类型列表
    Methods:
        from_json: 从JSON文件加载配置
        save_to_file: 保存配置到JSON文件
        update: 更新配置参数
        
    Example:
        >>> config = FrameConfig.from_json('config.json')
        >>> config.update(
        ...     paths={'output_dir': './new_output'},
        ...     scene_detection={'fast': {'threshold': 30.0}}
        ... )
        >>> config.save_to_file('updated_config.json')
    """

    # 默认配置
    output_dir: Path = field(default_factory=lambda: Path('./assets/keyframes'))
    video_path: Path = field(default_factory=lambda: Path('./assets/video'))

    # 保存设置
    save_image_format: str = 'jpg'
    save_frames: bool = True
    save_metadata: bool = True

    # 图像处理设置
    resize_frame: bool = True
    frame_size: Tuple[int, int] = (640, 480)

    # 场景检测配置
    scene_detection: Dict[str, SceneDetectionConfig] = field(default_factory=lambda: {
        'fast': SceneDetectionConfig(threshold=30, min_scene_len=0.3),
        'normal': SceneDetectionConfig(threshold=27, min_scene_len=0.5),
        'slow': SceneDetectionConfig(threshold=25, min_scene_len=1.0)
    })
    # 处理阈值
    quality_threshold: float = 0.6
    similarity_threshold: float = 0.85

    # 并行提取
    parallel_extract: bool = True 

    # 分类规则
    categories:Dict[str, List[str]] = field(default_factory=lambda: {
        'key_frames':['motion', 'object', 'quality'],
        'scene_changes':['content'],
        'quality_frames':['quality'],
        'regular_frames':['motion', 'object']
    })

    def __post_init__(self):
        """数据类初始化后的验证"""
        self._validate_config()

    def _validate_config(self)->None:
        """验证配置的有效性"""
        try:
            # 验证路径
            if not isinstance(self.output_dir, Path):
                self.output_dir = Path(self.output_dir)
            if not isinstance(self.video_path, Path):
                self.video_path = Path(self.video_path)
            # 验证阈值
            if not isinstance(self.save_image_format, str):
                raise ValueError(f"图像保存格式必须是字符串类")
            if not isinstance(self.save_frames, bool):
                raise ValueError(f"是否保存帧图像必须是布尔类型")
            if not isinstance(self.save_metadata, bool):
                raise ValueError(f"是否保存元数据必须是布尔类型")
            # 验证场景检测配置
            if not isinstance(self.scene_detection, dict):
                raise ValueError(f"场景检测配置必须是字典类型")
            for speed, config in self.scene_detection.items():
                if not isinstance(speed, str):
                    raise ValueError(f"场景速度类型必须是字符串")
                if not isinstance(config, SceneDetectionConfig):
                    raise ValueError(f"场景检测配置{speed}必须是SceneDetectionConfig类型")
                if not 0<= config.threshold <= 100:
                    raise ValueError(f"场景检测阈值必须在0到100之间")
                if not 0<= config.min_scene_len <= 10:
                    raise ValueError(f"场景{speed}检测最小场景长度必须在0到10秒之间")
            # 验证质量阈值
            if not isinstance(self.quality_threshold, (int,float)):
                raise ValueError(f"质量阈值必须是整数或浮点数类型") 
            if not 0<= self.quality_threshold <= 1:
                raise ValueError(f"质量阈值必须在0到1之间")
            # 验证相似度阈值
            if not isinstance(self.similarity_threshold, (int,float)):
                raise ValueError(f"相似度阈值必须是整数或浮点数类型")
            if not 0<= self.similarity_threshold <= 1:
                raise ValueError(f"相似度阈值必须在0到1之间")
            if not isinstance(self.parallel_extract, bool):
                raise ValueError(f"是否启用并行提取必须是布尔类型")
            
            # 验证分类规则
            if not isinstance(self.categories, dict):
                raise ValueError(f"分类规则必须是字典类型")
            for category, frame_types in self.categories.items():
                if not isinstance(category, str):
                    raise ValueError(f"分类类型必须是字符串")
                if not isinstance(frame_types, list):
                    raise ValueError(f"分类类型{category}的帧类型必须是列表")
                if not all(isinstance(frame_type, str) for frame_type in frame_types):
                    raise ValueError(f"分类类型{category}的帧类型必须是字符串列表")
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            raise ConfigError(f"配置验证失败: {e}")

    @classmethod
    def load_from_file(cls, config_path:Union[str, Path])->'FrameConfig':
        """从JSON文件加载配置
        
        Args:
            config_path: 配置文件路径f
            
        Returns:
            FrameConfig: 配置对象
            
        Raises:
            ConfigError: 配置加载或验证失败时抛出
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.error(f"配置文件不存在: {config_path},返回默认配置")
                return cls()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            required_keys = ['paths', 'save_settings', 'thresholds', 'settings', 'categories']
            for key in required_keys:
                if key not in config_dict:
                    raise ConfigError(f"配置文件缺少必要的键: {key}")
            
            scene_detection = {}
            for speed, config in config_dict['scene_detection'].items():
                scene_detection[speed] = SceneDetectionConfig(
                    threshold=config['content_threshold'],
                    min_scene_len=config['min_scene_len']
                )

            image_processing = config_dict.get('image_processing',{})
            processed_config = {
                'output_dir': Path(config_dict['paths']['output_dir']),
                'video_path': Path(config_dict['paths']['video_path']),
                'save_image_format': config_dict['save_settings']['save_image_format'],
                'save_frames': config_dict['save_settings']['save_frames'],
                'save_metadata': config_dict['save_settings']['save_metadata'],
                'scene_detection': scene_detection,
                'resize_frame': image_processing.get('resize_frame',True),
                'frame_size': tuple(image_processing.get('frame_size',[640,480])),
                'quality_threshold': config_dict['thresholds']['quality_threshold'],
                'similarity_threshold': config_dict['thresholds']['similarity_threshold'],
                'parallel_extract': config_dict['settings']['parallel_extract'],
                'categories': config_dict['categories']
            }
            return cls(**processed_config)
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}")
            raise ConfigError(f"配置文件格式错误: {e}")
        except KeyError as e:
            logger.error(f"配置文件缺少必要的键: {e}")
            raise ConfigError(f"配置文件缺少必要的键: {e}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            return cls()
        
    def save_to_file(self, config_path:Union[str, Path])->None:
        """保存配置到JSON文件
        
        Args:
            config_path: 配置文件保存路径
            
        Raises:
            ConfigError: 保存失败时抛出
        """
        try:
            config_path = Path(config_path)
            config_dict = {
                'paths': {
                    'output_dir': str(self.output_dir),
                    'video_path': str(self.video_path)
                },
                'save_settings': {
                    'save_image_format': self.save_image_format,
                    'save_frames': self.save_frames,
                    'save_metadata': self.save_metadata
                },
                'image_processing': {
                    'resize_frame': self.resize_frame,
                    'frame_size': list(self.frame_size)
                },
                'scene_detection': {
                     speed:{
                        'threshold':config.threshold,
                        'min_scene_len':config.min_scene_len
                    }
                    for speed, config in self.scene_detection.items()                    
                },
                'thresholds': {
                    'quality_threshold': self.quality_threshold,
                    'similarity_threshold': self.similarity_threshold
                },
                'settings': {
                    'parallel_extract': self.parallel_extract
                },
                'categories': self.categories
            }
            # 创建父目录
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4,ensure_ascii=False)            
            logger.info(f"配置文件保存成功: {config_path}")

        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
            raise ConfigError(f"保存配置文件失败: {str(e)}")
        
    def update(self, **kwargs)->None:
        """更新配置项
        
        支持嵌套结构的配置更新，例如：
        >>> config.update(
        ...     paths={'output_dir': './new_path'},
        ...     thresholds={'quality_threshold': 0.8}
        ... )
        
        Args:
            **kwargs: 要更新的配置项键值对
            
        Raises:
            ConfigError: 更新失败时抛出
        """ 
        try:
            # 更新路径配置
            if 'paths' in kwargs:
                paths = kwargs.pop('paths')
                if 'output_dir' in paths:
                    self.output_dir = Path(paths['output_dir'])
                if 'video_path' in paths:
                    self.video_path = Path(paths['video_path'])
            # 更新保存设置
            if 'save_settings' in kwargs:
                save_settings = kwargs.pop('save_settings')
                if 'save_image_format' in save_settings:
                    self.save_image_format = save_settings['save_image_format']
                if 'save_frames' in save_settings:
                    self.save_frames = save_settings['save_frames']
                if 'save_metadata' in save_settings:
                    self.save_metadata = save_settings['save_metadata']
            # 更新场景检测配置
            if 'scene_detection' in kwargs:
                scene_detection = kwargs.pop('scene_detection')
                for speed, config in scene_detection.items():
                    if speed in self.scene_detection:
                        current_config = self.scene_detection[speed]
                        if 'threshold' in config:
                            current_config.threshold = float(config['threshold'])
                        if 'min_scene_len' in config:
                            current_config.min_scene_len = float(config['min_scene_len'])
                    else:
                        self.scene_detection[speed] = SceneDetectionConfig(
                            threshold=float(config.get('threshold',27.0)),
                            min_scene_len=float(config.get('min_scene_len',0.5))
                        )
            # 更新处理阈值
            if 'thresholds' in kwargs:
                thresholds = kwargs.pop('thresholds')
                if 'quality_threshold' in thresholds:
                    self.quality_threshold = thresholds['quality_threshold']
                if 'similarity_threshold' in thresholds:
                    self.similarity_threshold = thresholds['similarity_threshold']
            # 更新并行提取设置
            if 'settings' in kwargs:
                settings = kwargs.pop('settings')
                if 'parallel_extract' in settings:
                    self.parallel_extract = settings['parallel_extract']
            # 更新分类规则
            if 'categories' in kwargs:
                self.categories = kwargs['categories']

            for key, value in kwargs.items():
                if hasattr(self, key):  
                    setattr(self, key, value)
                else:
                    logger.warning(f"配置项不存在: {key}")
            self._validate_config()
        except Exception as e:
            logger.error(f"更新配置失败: {str(e)}")
            raise ConfigError(f"更新配置失败: {str(e)}")
    
    def to_dict(self)->dict[str, Any]:
        """将配置转换为字典"""
        return asdict(self)

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
            logger.error(f"帧数据验证失败: {e}")
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
                logger.error("帧图像数据为空，无法保存")
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
                logger.error(f"帧图像数据类型错误: {type(self.image)}")
                return False
            
            success = cv2.imwrite(str(image_path), self.image)

            if not success:
                logger.error(f"保存帧图像失败: {image_path}")
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
                    
            logger.info(f"保存帧成功: {image_path} (类别: {category or 'unknown'})")
            return True
        except Exception as e:
            logger.error(f"保存帧失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

class KeyFrameExtractor(ABC):
    """关键帧提取器基类
    
    职责：
    1. 提供统一的提取器接口
    2. 管理视频资源的生命周期
    3. 提供基础的视频处理功能
    4. 标准化帧的创建和验证
    
    Attributes:
        config: 配置对象
        video_capture: 视频捕获对象
        fps: 视频帧率
        frame_count: 视频总帧数
        current_frame_idx: 当前帧索引
    """

    def __init__(self,config:FrameConfig):
        """初始化提取器
        
        Args:
            config: 配置参数字典
        """

        if config is None:
            raise ValueError("配置参数不能为空")
        self.config = config
        self.video_capture = None
        self.fps = 0
        self.frame_count = 0
        self.current_frame_idx = 0
        self._setup_logger()
    
    def _setup_logger(self)->None:
        """设置日志记录器
        
        功能：
        1. 创建类专属的日志记录器
        2. 设置日志级别
        3. 配置控制台和文件输出
        4. 自定义日志格式
        5. 添加上下文信息
        """
        try:    
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.DEBUG)

            if self.logger.handlers:
                return
            
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            log_file = log_dir / f"{self.__class__.__name__}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)

            error_log_file = log_dir / f"{self.__class__.__name__}_error.log"
            error_file_handler = logging.FileHandler(error_log_file, encoding='utf-8')
            error_file_handler.setLevel(logging.ERROR)

            formatter = logging.Formatter('[%(asctime)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
                                datefmt = '%Y-%m-%d %H:%M:%S')

            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            error_file_handler.setFormatter(formatter)


            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(error_file_handler)

            self.logger.info(f"初始化{self.__class__.__name__}日志记录器成功")
            self.logger.debug(f"日志文件路径：{log_file}")
            self.logger.debug(f"错误日志文件路径：{error_log_file}")
        except Exception as e:
            print(f"设置日志记录器失败: {str(e)}")
            raise 

    @abstractmethod
    def extract(self, video_path:Union[str, Path])->list[Frame]:
        """提取关键帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[Frame]: 提取的关键帧列表
        """
        pass

    def load_video(self, video_path:Union[str, Path])->bool:
        """加载视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                self.logger.error(f"视频文件不存在: {video_path}")
                raise ValueError(f"视频文件不存在: {video_path}")
            self.video_capture = cv2.VideoCapture(str(video_path))
            if not self.video_capture.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_idx = 0

            self.logger.info(f"成功加载视频文件: {video_path}")
            self.logger.info(f"FPS: {self.fps}, 总帧数: {self.frame_count}")
            return True
        except Exception as e:
            self.logger.error(f"加载视频文件失败: {e}")
            self.release_video()
            return False
    
    def preprocess_frame(self, frame: np.ndarray)->Optional[np.ndarray]:
        """预处理帧
        
        Args:
            frame: 输入帧
            
        Returns:
            Optional[np.ndarray]: 预处理后的帧
        """
        
        try:
            if frame is None:
                self.logger.error("输入帧为空")
                return None
            self.logger.debug(f"开始预处理帧，原始形状 {frame.shape}")

            if hasattr(self.config, 'resize_frame') and self.config.resize_frame:
                self.logger.debug(f"调整帧大小为：{self.config.frame_size}")
                if hasattr(self.config, 'frame_size'):
                    frame = cv2.resize(frame, self.config.frame_size)
                    self.logger.debug(f"调整后帧形状：{frame.shape}")    
            else:
                self.logger.debug(f"不调整帧大小")
            if frame is not None and frame.size >0:
                self.logger.debug("预处理帧成功")        
                return frame
            else:
                self.logger.error("预处理帧数据无效")
                return None

        except Exception as e:
            self.logger.error(f"预处理帧失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def read_frame(self)->Optional[np.ndarray]:
        """读取下一帧
        
        Returns:
            Optional[np.ndarray]: 帧数据, 如果到达视频末尾则返回None
        """
        if self.video_capture is None:
            return None
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame_idx +=1
            return frame
        return None
    
    def create_frame(self, image:np.ndarray, 
                     metadata:Optional[Dict[str,Any]]=None,
                     frame_index:Optional[int]=None,
                     timestamp:float=0.0)->Optional[Frame]:
        """创建标准化的帧对象
        
        Args:
            image: 帧图像数据
            metadata: 额外的元数据
            
        Returns:
            Optional[Frame]: 帧对象
        """
        try:
            if image is None:
                self.logger.error("帧图像数据为空")
                return None
            current_idx = frame_index if frame_index is not None else self.current_frame_idx
            current_timestamp = timestamp if timestamp is not None else (current_idx / self.fps if self.fps >0 else 0)
            base_metadata = {
                'extractor_type':self.__class__.__name__,
                'frame_index': current_idx,
                'timestamp': current_timestamp,
                'extraction_time': datetime.now().isoformat(),
            }
            if metadata:
                base_metadata.update(metadata)
            frame = Frame(image=image, 
                         metadata=base_metadata,
                         frame_index=current_idx,
                         timestamp=current_timestamp)
            self.logger.debug(f"创建帧对象成功: {frame},index:{frame.frame_index},timestamp:{frame.timestamp}")
            return frame
        
        except Exception as e:
            self.logger.error(f"创建帧对象失败: {str(e)}")
            return None
        
    def release_video(self)->None:
        """释放视频资源"""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            self.current_frame_idx = 0

    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release_video()




class ContentDetectorExtractor(KeyFrameExtractor):
    """基于内容的场景切换检测器
    
    使用场景检测库的ContentDetector来检测视频中的场景切换
    """
    def __init__(self,config:FrameConfig):
        """初始化内容检测器

        使用 PySceneDetect 的 ContentDetector 进行场景切换检测，
        提取场景变化的关键帧。
        
        Args:
            config: 配置参数，可包含:
                - threshold: 检测阈值 (默认: 27.0)
                - min_scene_len: 最小场景长度 (默认: 15)
                - luma_only: 是否只使用亮度信息 (默认: False)
        """

        super().__init__(config)
        self.detection_modes = ['fast','normal','slow']
        self.detector = None
        self._init_detector('fast')

    def _init_detector(self,mode:str):
        """初始化指定模式的检测器
        
        Args:
            mode: 检测模式 ('fast', 'normal', 'slow')
        """
        scene_config = self.config.scene_detection.get(mode,SceneDetectionConfig (
            threshold=27.0,
            min_scene_len=0.5
        ))
        self.current_mode = mode
        self.threshold = scene_config.threshold
        self.min_scene_len = scene_config.min_scene_len
        self.detector = ContentDetector(
            threshold=self.threshold,
            min_scene_len=self.min_scene_len
        )
        self.logger.info(f"初始化{mode}模式检测器成功:阈值{self.threshold},最小场景长度{self.min_scene_len}")
        
    def extract(self, video_path:str)->list[Frame]:
        """提取场景切换的关键帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[Frame]: 检测到的关键帧列表
        """
        try:
            if not self.load_video(video_path):
                self.logger.error("加载视频文件失败")
                return []
            
            # 存储所有关键帧
            all_keyframes = []
            mode_index = 0

            self.logger.info("尝试读取第一帧...")
            ret, first_frame = self.video_capture.read()
            if ret and first_frame is not None:
                self.logger.info(f"读取第一帧成功。帧形状：{first_frame.shape}")
                processed_first_frame = self.preprocess_frame(first_frame)
                if processed_first_frame is not None:
                    first_frame_obj = self.create_frame(
                        image=processed_first_frame,
                        metadata={'method':'content_detection',
                                  'detection_mode':'first_frame',
                                  'scene_idx': 0,  
                                  'note':'视频第一帧',                        
                                 'threshold': self.threshold,                                
                        }
                    )
                    if first_frame_obj:
                        all_keyframes.append(first_frame_obj)
                        self.logger.info("成功添加第一帧到关键帧列表")
                else:
                    self.logger.warning("预处理第一帧失败,尝试使用原始帧")
                    first_frame_obj = self.create_frame(
                        image=first_frame,
                        metadata={'method':'content_detection',
                                  'detection_mode':'first_frame',
                                  'scene_idx': 0,  
                                  'note':'视频第一帧(原始帧)',                        
                                  'threshold': self.threshold,                                
                        },
                        frame_index=start_frame,
                        timestamp=timestamp
                    )
                    if first_frame_obj:
                        all_keyframes.append(first_frame_obj)
                        self.logger.info("成功添加第一帧(原始帧）到关键帧列表")
            # 重置视频捕获对象到第一帧
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while mode_index < len(self.detection_modes):
                mode = self.detection_modes[mode_index]
                self.logger.info(f"初始化{mode}模式检测器...")
                
                self._init_detector(mode)
                self.logger.info(f"尝试使用{mode}模式检测器...")

                try: 
                    scene_list = detect(video_path=str(video_path), detector=self.detector)
                
                    if scene_list:  
                        mode_keyframes = []          
                        for scene_idx, scene in enumerate(scene_list):
                            start_frame = scene[0].get_frames()
                            if start_frame == 0:
                                continue
                            timestamp = start_frame / self.fps if self.fps else 0
                            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                            frame = self.read_frame()
                            if frame is not None:
                                processed_frame = self.preprocess_frame(frame)
                                if processed_frame is None:
                                    continue

                                frame_obj = self.create_frame(
                                    image=processed_frame,
                                    frame_index=start_frame,
                                    timestamp=timestamp,
                                    metadata={'method':'content_detection',
                                            'scene_idx': scene_idx,                          
                                            'confidence': self._calculate_confidence(frame),
                                            'scene_length': scene[1].get_seconds() - scene[0].get_seconds(),
                                            'detection_mode': mode,
                                    }
                                )
                                if frame_obj:
                                    mode_keyframes.append(frame_obj)
                                self.logger.info(
                                    f"检测到场景切换:场景{scene_idx}, "
                                    f"帧索引:{start_frame}, "
                                    f"时间戳:{frame_obj.timestamp:.2f}s"
                                )
                        if mode_keyframes:
                            self.logger.info(f"成功检测到{len(mode_keyframes)}个场景切换关键帧")
                            all_keyframes.extend(mode_keyframes)                        
                            break
                    else:
                        self.logger.info(f"尝试使用{mode}模式检测器失败,进入下一个模式...")
                        mode_index += 1   
                except Exception as e:
                    self.logger.error(f"使用{mode}模式检测器失败: {str(e)}")
                    mode_index += 1
                    continue
            total_frames = len(all_keyframes)
            if total_frames == 0:
                self.logger.warning("没有检测到任何场景切换关键帧")
            elif total_frames == 1:
                self.logger.info("检测到1个场景切换关键帧")
            else:
                self.logger.info(f"总共提取了{total_frames}个场景切换关键帧包括第一帧")
                self.logger.debug("关键帧详情")
                for idx, frame in enumerate(all_keyframes):
                    self.logger.debug(f"帧索引:{frame.frame_index},时间戳:{frame.timestamp:.2f}s")
            
            return all_keyframes
        
        except Exception as e:
            self.logger.error(f"内容检测失败: {str(e)}")    
            import traceback
            traceback.print_exc()
            return []
        
        finally:
            self.release_video()

    def _calculate_confidence(self, frame: np.ndarray) -> float:
        """计算帧的置信度分数
        
        Args:
            frame: 输入帧
            
        Returns:
            float: 置信度分数 (0-1)
        """
        try:
            if frame is None:
                return 0.0
                
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算清晰度 (使用Laplacian算子)
            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
            clarity_score = min(1.0, clarity / 500.0)
            
            # 计算亮度
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # 计算对比度
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 64.0)
            
            # 综合评分
            confidence = (
                clarity_score * 0.4 +
                brightness_score * 0.3 +
                contrast_score * 0.3
            )
            
            self.logger.debug(f"帧置信度计算 - 清晰度: {clarity_score:.2f}, "
                            f"亮度: {brightness_score:.2f}, "
                            f"对比度: {contrast_score:.2f}, "
                            f"最终分数: {confidence:.2f}")
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"计算置信度失败: {e}")
            return 0.0

# class AdaptiveDetectorExtractor(KeyFrameExtractor):
    # """自适应场景切换检测器 
        
    # 使用场景检测库的AdaptiveDetector来检测视频中的场景切换
    # 相比ContentDetector, 这个检测器能更好地处理渐变场景
    # """
    # def __init__(self,config:Dict[str, Any]=None):
    #     """初始化自适应检测器
        
    #     Args:
    #         config: 配置参数，可包含:
    #             - min_scene_len: 最小场景长度 (默认: 15)
    #             - window_width: 分析窗口宽度 (默认: 2)
    #             - luma_only: 是否只使用亮度信息 (默认: False)
    #             - min_content_val: 最小内容值 (默认: 15.0)
    #             - weights: 颜色通道权重 (默认: (1.0, 1.0, 1.0))
    #             - adaptive_threshold: 自适应阈值 (默认: 3.0)
    #     """
    #     super().__init__(config)
    #     self.default_config = {
    #         'min_scene_len': 3,
    #         'luma_only': False,
    #         'min_content_val': 15,
    #         'window_width': 2,
    #     }
    #     self.config = {**self.default_config, **(config or {})}
    #     self.detector = None
    #     print("adaptive detector start.....")

    # def _init_detector(self):
    #     """初始化自适应检测器"""
    #     self.detector = AdaptiveDetector(
    #         min_scene_len=self.config['min_scene_len'],
    #         luma_only=self.config['luma_only'],
    #         window_width=self.config['window_width'],
    #         min_content_val=self.config['min_content_val'],
    #     )
    
    # def get_config(self,config:Dict[str, Any])->bool:
    #     """设置配置参数
        
    #     Args:
    #         config: 新的配置参数
            
    #     Returns:
    #         bool: 设置是否成功
    #     """
    #     try:
    #         self.config.update(config)
    #         return True
    #     except Exception as e:
    #         print(f"更新配置失败: {e}")
    #         return False
    
    # def set_config(self, config: Dict[str, Any]) -> bool:
    #     return super().set_config(config)

    # def extract(self, video_path:str)->list[Frame]:
    #     """提取关键帧
        
    #     Args:
    #         video_path: 视频文件路径
            
    #     Returns:
    #         List[Frame]: 检测到的关键帧列表
    #     """
    #     try:
    #         self._init_detector()
    #         super()._init_video_info(video_path)
            
    #         scenes = detect(video_path, self.detector)
    #         if not scenes:
    #             print(f"没有检测到场景, 尝试降低阈值")
    #             return []

    #         keyframes = []
    #         cap = cv2.VideoCapture(str(video_path))
    #         for scene_idx, scene in enumerate(scenes):
    #             start_frame = scene[0].get_frames()
    #             end_frame = scene[1].get_frames()
    #             scene_length = end_frame - start_frame
    #             timestamp = start_frame / self.video_info['fps']
    #             #主要目的是跳转到目标帧，而不需要从头开始
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    #             ret, frame = cap.read()
    #             if not ret:
    #                 continue

    #             scene_score = self._calculate_scene_score(frame)
    #             transition_type = self._detect_transition_type(
    #                 cap, start_frame, end_frame
    #             )

    #             keyframe = Frame(   
    #                 image=frame,
    #                 frame_index=start_frame,
    #                 timestamp=timestamp,
    #                 metadata={
    #                     'scene_idx': scene_idx,
    #                     'method':'adaptive_detection',
    #                     'confidence': scene_score,
    #                     'transition_type': transition_type,
    #                     'scene_length': scene_length,
    #                     'fps': self.video_info['fps']
    #                 }
    #             )
    #             keyframes.append(keyframe)

    #             if self.config['save_keyframes']:
    #                 self._save_keyframes(keyframe, scene_idx, video_path, "adaptive")
    #         cap.release()
    #         print(f"检测到{len(keyframes)}个场景切换")
            
    #         return keyframes
    #     except Exception as e:
    #         print(f"自适应检测失败: {e}")
    #         return []
        
    # def _calculate_scene_score(self, frame:np.ndarray)->float:
    #     """计算场景分数
        
    #     基于图像的多个特征计算场景重要性分数
    #     """
    #     try:
    #         # 人肉眼看得清楚的，有意义的关键帧的提取，将模糊的人眼看不清楚的给忽略掉
    #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #         # 亮度
    #         brightness = np.mean(hsv[:,:,2])
    #         # 对比度
    #         contrast = np.std(hsv[:,:,2])
    #         # 饱和度
    #         saturation = np.mean(hsv[:,:,1])

    #         score = (brightness / 255 * 0.4 +
    #                  contrast / 255 * 0.4 +
    #                  saturation /255 * 0.2)
    #         return float(min(1.0, score))
        
    #     except Exception as e:
    #         print(f"计算场景分数失败: {e}")
    #         return 0.0

    # def _detect_transition_type(self, cap:cv2.VideoCapture, start_frame:int, end_frame:int)->str:
    #     """检测场景转换类型
        
    #     Args:
    #         cap: 视频捕获对象
    #         start_frame: 开始帧
    #         end_frame: 结束帧
            
    #     Returns:
    #         str: 转换类型 ('cut', 'fade', 'dissolve', 'unknown')
    #     """
    #     try:
    #         sample_frame = []
    #         for frame_idx in np.linspace(start_frame, end_frame, 5):
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    #             ret, frame = cap.read()
    #             if ret:
    #                 sample_frame.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    #         if len(sample_frame)<2:
    #             return 'unknown'
            
    #         diffs = []
    #         for i in range(len(sample_frame)-1):
    #             diff = np.mean(np.abs(sample_frame[i+1].astype(float) 
    #                                   - sample_frame[i].astype(float)))
    #             diffs.append(diff)
            
    #         max_diff = max(diffs)
    #         # 差异大于50，认为是突变
    #         if max_diff > 50:
    #             return 'cut'
    #         elif np.all(np.array(diffs) > 0): # np.all()差异都大于0，持续增加
    #             return 'fade'   
    #         elif np.mean(diffs) > 20: # 差异均值大于20，认为是渐变
    #             return 'dissolve'
    #         else:
    #             return 'unknown'
    #     except Exception as e:
    #         print(f"检测场景转换类型失败: {e}")
    #         return 'unknown'

# class IntervalDetectorExtractor(KeyFrameExtractor):
    # """基于固定间隔的关键帧提取器
    
    # 按照固定的时间或帧数间隔提取关键帧，
    # 并对提取的帧进行质量评估
    # """
    # def __init__(self, config:Dict[str, Any]=None):
    #     """初始化间隔提取器
        
    #     Args:
    #         config: 配置参数，可包含:
    #             - interval_sec: 时间间隔（秒）
    #             - interval_frames: 帧间隔（优先级低于时间间隔）
    #             - min_quality_score: 最小质量分数
    #             - save_keyframes: 是否保存关键帧
    #             - quality_check: 是否进行质量检查
    #     """
    #     super().__init__(config)
    #     self.default_config = {
    #         'interval_sec': 0.5,
    #         'interval_frames': 30,
    #         'min_quality_score': 0.5,
    #         'save_keyframes': True,
    #         'quality_check': True
    #     }
    #     self.config = {**self.default_config, **(config or {})}
    #     self.video_info = None
    #     print("interval detector start.....")
        
    # def extract(self, video_path:str)->list[Frame]:
    #     """提取关键帧"""
    #     try:
    #         super()._init_video_info(video_path)
    #         if not self.video_info:
    #             raise ValueError("视频信息未初始化")
    #         cap = cv2.VideoCapture(str(video_path))
            
    #         if 'interval_sec' in self.config:
    #             interval = int(self.video_info['fps'] * self.config['interval_sec'])
    #         else:
    #             interval = self.config['interval_frames']
    #         keyframes = []
    #         frame_count = 0
    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #             # 如果当前帧数是提取间隔的整数倍，则提取关键帧
    #             if frame_count % interval == 0:
    #                 quality_score = self._calculate_quality_score(frame)
    #                 # 如果不需要质量检查，或者质量分数大于等于最小质量分数，则提取关键帧
    #                 if quality_score >= self.config.get('min_quality_score', 0.5):
    #                     timestamp = frame_count / self.video_info['fps']
    #                     keyframe = Frame(
    #                         image=frame,
    #                         frame_index=frame_count,
    #                         timestamp=timestamp,
    #                         metadata={
    #                             'quality_score': quality_score,
    #                             'method': 'interval',
    #                             'interval': interval,
    #                             'fps': self.video_info['fps']
    #                         }
    #                     )
    #                     keyframes.append(keyframe)  
    #                     if self.config['save_keyframes']:
    #                         self._save_keyframes(keyframe, frame_count, video_path, "interval")
    #                 frame_count += 1
    #         cap.release()
    #         print(f"提取了{len(keyframes)}个关键帧")
    #         return keyframes             
    #     except Exception as e:
    #         print(f"固定间隔提取失败: {e}")
    #         print(f"错误详情: {str(e)}")
    #         print(f"video_info: {self.video_info}")
    #         return []
    
    

    # def _evaluate_frame_quality(self, frame:np.ndarray)->float:
    #     """评估帧的质量
        
    #     评估指标包括:
    #     - 清晰度
    #     - 亮度均衡
    #     - 对比度
    #     """
    #     try:
    #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #         # 清晰度
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
    #         clarity_score = min(1.0, clarity/500.0)

    #         # 亮度均衡
    #         brightness = np.mean(hsv[:,:,2])
    #         brightness_score = 1.0- abs(brightness -128) / 128

    #         # 对比度
    #         contrast = np.std(hsv[:,:,2])
    #         contrast_score = min(1.0, contrast/64.0)

    #         quality_score = (
    #             clarity_score * 0.4 +
    #             brightness_score * 0.3 +
    #             contrast_score * 0.3
    #         )

    #         return float(quality_score)

    #     except Exception as e:
    #         print(f"评估帧质量失败: {e}")
    #         return 0.0

# 运动检测
# class MotionDetectionExtractor(KeyFrameExtractor):
    
    # def extract(self, vido_path:str)->list[Frame]:
    #     pass

    
    # def update_config(self, new_config:Dict[str, Any])->None:
    #     """更新配置"""
    #     def _recursive_update(base_dict:dict, update_dict:dict)->None:
    #         for key, value in update_dict.items():
    #             if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    
    #                 _recursive_update(base_dict[key], value)
    #             else:
    #                 base_dict[key] = value
    #     _recursive_update(self.config, new_config)

class FrameProcessor:
    """帧处理器：用于合并和去重关键帧"""
    def merge_results(self, extractor_results:Dict[str, List[Frame]])->List[Frame]:
        """合并多个提取器的结果
        
        Args:
            extractor_results: 各提取器的结果字典
            
        Returns:
            List[Frame]: 合并后的帧列表（按时间戳排序）
        """
        merged_frames = []
        for frames in extractor_results.values():
            # 将每个提取器的帧结果添加到合并帧列表中
            # frames 是一个包含帧对象的列表
            # merged_frames 是一个合并后的帧列表
            # extend 方法用于将 frames 列表中的所有元素添加到 merged_frames 列表中
            # 这样可以将所有提取器的结果合并到一个列表中
            merged_frames.extend(frames)
        merged_frames.sort(key=lambda x: x.timestamp)
        return merged_frames
    
    def remove_duplicates(self, frames:List[Frame],
                          similarity_threshold:float=0.85,
                          time_threshold:float=0.5)->List[Frame]:
        """去除重复帧
        
        Args:
            frames: 输入帧列表
            similarity_threshold: 相似度阈值
            time_threshold: 时间阈值（秒）
            
        Returns:
            List[Frame]: 去重后的帧列表
        """
        if not frames:
            return []
        
        unique_frames = [frames[0]]
        for frame in frames[1:]:
            if frame.timestamp - unique_frames[-1].timestamp > time_threshold:
                # 如果时间差大于时间阈值，则需要计算相似度
                is_similar = False
                for unique_frame in unique_frames:
                    similarity = self._calculate_similarity(frame, unique_frame)
                    if similarity >= similarity_threshold:
                        is_similar = True
                        break
                if not is_similar:
                    unique_frames.append(frame)
            else:
                # 如果时间差小于时间阈值，则直接添加
                unique_frames.append(frame)                
        return unique_frames
    
    def _calculate_similarity(self, frame1:Frame, frame2:Frame)->float:
        """计算两帧之间的相似度
        
        Args:
            frame1: 第一帧
            frame2: 第二帧
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            img1 = cv2.resize(frame1.image, (64,64))
            img2 = cv2.resize(frame2.image, (64,64))

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            hist1 = cv2.calcHist([gray1], [0], None, [256], [0,256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0,256])

            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)

            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            return max(0, similarity)
    
        except Exception as e:
            print(f"计算相似度失败: {e}")
            return 0.0
        
class ExtractorFactory(Protocol):
    def create(self, conifg:FrameConfig)->'KeyFrameExtractor':
        pass
    
class ContentDetectorFactory:
    def create(self, conifg:FrameConfig)->'ContentDetectorExtractor':
        return ContentDetectorExtractor(config=conifg)
    
class KeyFrameManager:
    """关键帧提取管理器
    
    统一管理所有关键帧提取器，整合输出结果
    """
    def __init__(self, config:FrameConfig, extractors:Dict[str, ExtractorFactory]):
        """
        初始化管理器
        """
        self.config = config
        self.extractors = extractors
        self.frame_processor = FrameProcessor()


    def process_video(self, video_path:str)->List[Frame]:
        """处理视频并整合所有提取器的结果
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[Frame]: 整合后的关键帧列表
        """
        try:    
            extractor_results = {}
            for name, factory in self.extractors.items():
                extractor_output = Path(self.config.output_dir) / name
                extractor_output.mkdir(parents=True, exist_ok=True)

                extractor = factory.create(self.config)
                frames = extractor.extract(video_path)
                if frames:
                    self._save_extractor_results(frames, extractor_output, name)
                    extractor_results[name] = frames
            if not extractor_results:
                print("没有提取到任何关键帧")
                return []
            # 合并结果
            merged_frames = self.frame_processor.merge_results(extractor_results)
            # 去重
            unique_frames = self.frame_processor.remove_duplicates(merged_frames)
            
            # 保存最终结果
            final_output = self.config.output_dir / 'final'
            final_output.mkdir(parents=True, exist_ok=True)
            self._save_final_results(unique_frames, final_output)
        
            return unique_frames
        except Exception as e:
            print(f"处理视频失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _save_extractor_results(self, frames:List[Frame], output_dir:Path, extractor_name:str)->None:
        """保存单个提取器的结果"""
        try:
            for i, frame in enumerate(frames):
                frame_name = f"{extractor_name}_{i:04d}"
                frame.save_frame(config=self.config, frame_name=frame_name, category=extractor_name)

            if self.config.save_metadata:   
                metadata_path = output_dir / f"{frame_name}.json"
                metadata = {
                    'extractor': extractor_name,
                    'frame_index': frame.frame_index,
                    'timestamp': frame.timestamp,
                    'path': str(metadata_path),
                }
                frame.add_metadata('save_info',metadata)

        except Exception as e:
            logger.error(f"保存单个提取器结果失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_final_results(self, frames:List[Frame], output_dir:Path)->None:
        """保存最终结果"""
        try:
            for i, frame in enumerate(frames):
                frame_name = f"final_{i:04d}"
                frame.save_frame(config=self.config, frame_name=frame_name,category='final')
                if self.config.save_metadata:
                    metadata_path = output_dir / f"{i:04d}.json"
                    metadata = {
                        'path': str(metadata_path),
                        'category': 'final', 
                        'frame_index': frame.frame_index,
                        'timestamp': frame.timestamp,
                    }
                    frame.add_metadata('save_info',metadata)
        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")
            import traceback
            traceback.print_exc()
    

def process_video(video_path:Union[str, Path], output_dir:Union[str, Path])->List[Frame]:
    """处理视频并提取关键帧
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录路径，默认为"./output"
        
    Returns:
        List[Frame]: 处理后的关键帧列表
        
    Example:
        >>> frames = process_video("videos/test.mp4", "output/keyframes")
    """
    try:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        config_path =Path("config.json")
        if config_path.exists():
            config = FrameConfig.load_from_file(config_path)
            config.update(paths={'output_dir': str(output_dir)})
        else:
            config = FrameConfig(
                output_dir=output_dir,
                video_path=video_path,
                save_frames=True,
                save_metadata=True,
                parallel_extract=True,
                scene_detection={
                    'fast':SceneDetectionConfig(
                        threshold=30.0,
                        min_scene_len=0.3,
                    ),
                    'normal':SceneDetectionConfig(
                        threshold=27.0,
                        min_scene_len=0.5,
                    ),
                    'slow':SceneDetectionConfig(
                        threshold=23.0,
                        min_scene_len=1,
                    )
                }
            )

        extractors = {  
            'content': ContentDetectorFactory(),
        }

        manager = KeyFrameManager(config, extractors)
        print(f"处理视频: {video_path}")
        frames = manager.process_video(video_path)
        print(f"提取了{len(frames)}个关键帧")
        if frames:
            print(f"处理完成，共提取了{len(frames)}个关键帧")
            print(f"结果保存在：{output_dir}")
        else:
            print("处理失败，没有提取到任何关键帧")
        return frames
    except Exception as e:
        print(f"处理视频失败: {str(e)}")
        return []

    


# 物体检测
# class ObjectDetectionExtractor(KeyFrameExtractor):
    def get_config(self)->Dict[str, Any]:
        return self.config
    
    def set_config(self, config:Dict[str, Any])->bool:
        try:
            self.config.update(config)
            return True
        except Exception as e:
            print(f"更新配置失败: {e}")
            return False
    def extract(self, vido_path:str)->list[Frame]:
        pass
# 质量检测
# class QualityDetectionExtractor(KeyFrameExtractor):
    def get_config(self)->Dict[str, Any]:
        return self.config
    
    def set_config(self, config:Dict[str, Any])->bool:
        try:
            self.config.update(config)
            return True
        except Exception as e:
            print(f"更新配置失败: {e}")
            return False
    def extract(self, vido_path:str)->list[Frame]:
        pass
# 聚类检测
# class ClusteringExtractor(KeyFrameExtractor):
    def get_config(self)->Dict[str, Any]:
        return self.config
    
    def set_config(self, config:Dict[str, Any])->bool:
        try:
            self.config.update(config)
            return True
        except Exception as e:
            print(f"更新配置失败: {e}")
            return False
    def extract(self, vido_path:str)->list[Frame]:
        pass
    
if __name__ == "__main__":
    video_path = "./assets/video/06.mp4"
    output_dir = "./assets/video/keyframes"

    try:
        frames = process_video(video_path, output_dir)
        if frames:
            print("\n关键帧时间戳:")
            for i, frame in enumerate(frames):
                print(f"{i}: {frame.timestamp:.2f}s")
    except Exception as e:
        print(f"处理视频失败: {str(e)}")

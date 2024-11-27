from abc import ABC, abstractmethod
from ..models import Frame
from ..bridge import LoggerBridge

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
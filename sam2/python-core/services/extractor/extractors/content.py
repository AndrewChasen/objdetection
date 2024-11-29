from dataclasses import dataclass
import cv2
from scenedetect import detect, ContentDetector
import numpy as np
from .base import KeyFrameExtractor
from models import Frame
from app import AppInitializer

class ConfigError(Exception):
    """配置错误异常类"""
    pass


@dataclass
class SceneDetectionConfig:
    """场景检测配置类"""
    threshold:float
    min_scene_len: float
class ContentDetectorExtractor(KeyFrameExtractor):
    """基于内容的场景切换检测器
    
    使用场景检测库的ContentDetector来检测视频中的场景切换
    """
    def __init__(self):
        """初始化内容检测器

        使用 PySceneDetect 的 ContentDetector 进行场景切换检测，
        提取场景变化的关键帧。
        
        Args:
            config: 配置参数，可包含:
                - threshold: 检测阈值 (默认: 27.0)
                - min_scene_len: 最小场景长度 (默认: 15)
                - luma_only: 是否只使用亮度信息 (默认: False)
        """
        app = AppInitializer.get_instance()
        self.frame_config = app.frame_bridge
        self.logger = app.logger

        super().__init__(self.frame_config)
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
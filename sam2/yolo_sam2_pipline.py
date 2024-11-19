import torch
import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from supervision.assets import download_assets, VideoAssets
import base64
import supervision as sv
from pathlib import Path
from scenedetect import detect, ContentDetector,AdaptiveDetector
from scenedetect.scene_manager import save_images

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


# # 获取视频的帧生成器，用于生成视频的每一帧
# frames_generator = sv.get_video_frames_generator(SOURCE_VIDEO, START_IDX, END_IDX)

# # 上下文管理器，用于保存图片
# images_sink = sv.ImageSink(
#     # 放在原来图片的父目录下
#     target_dir_path=VIDEO_FRAMES_DIR.as_posix(),
#     overwrite=True,
#     image_name_pattern="image_{:05d}.png"
# )
# frame_counter = 0
# # 上下文管理器，用于保存图片
# with images_sink:
#     for frame in frames_generator:
#         frame = sv.scale_image(frame, SCALE_FACTOR)
#         images_sink.save_image(frame)
#         frame_counter += 1
#         if frame_counter == 10:
#             print(f"frame_counter: {frame_counter} done...")

# TARGET_VIDEO= f"{Path(VIDEO_FRAMES_DIR).stem}_result.mp4"
# # 获取图片的路径，并按照文件名排序
# SOURCE_FRAME_PATHS = sorted(sv.list_files_with_extensions(VIDEO_FRAMES_DIR.as_posix(),extensions=["jpeg"]))

# print(f"done")


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

# 选择关键帧，我们使用第一种方法，通过传统方法来确定关键帧
# '提取方法': {
#             '图像差异': '检测动作变化',
#             '特征提取': '检测内容变化',
#             '场景检测': '检测场景转换',
#             'CLIP分析': '检测语义变化'
#         },
# 选择关键帧，我们使用第二种方法，通过论文中的理论来训练模型来确定关键帧 todo！！！
# 选择关键帧，我们使用第三种方法，将两者结合起 todo！！！

#第一步，先使用scenedetect库来确定关键帧，如果不理想，再添加clip分析
class KeyFrameExtractor:
    def __init__(self, video_path, output_dir, config=None):
        """
        关键帧提取器
        Args:
            video_path: 视频路径
            output_dir: 输出目录
            config: 配置参数字典
        """
        self.config = config or {
            "content_threshold": 27.0,  # 提高阈值以减少误检
            "min_scene_len": 3,  # 增加最小场景长度
            "interval": 0.5,     # 调整间隔
            "adaptive_threshold": 3.0,  # 自适应检测阈值
            "luma_only": False,   # 是否只使用亮度通道
        }
        
        try:    
            self.video_path = Path(video_path)
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            if not self.video_path.exists():
                raise FileNotFoundError(f"Video file not found: {self.video_path}")
            if not self.video_path.suffix in ['.mp4', '.avi', '.mov']:
                raise ValueError(f"Unsupported video format: {self.video_path.suffix}")
            self._init_video_info()
        except Exception as e:
            print(f"Error initializing KeyFrameExtractor: {str(e)}")

    def _init_video_info(self):
        cap = cv2.VideoCapture(str(self.video_path))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count / self.fps
        cap.release()


    def _save_keyframes(self, scenes, method_name):
        """
        优化的关键帧保存函数
        """
        try:
            keyframe_paths = []
            output_dir = self.output_dir / method_name
            output_dir.mkdir(exist_ok=True)
            
            cap = cv2.VideoCapture(str(self.video_path))
            
            # 优化：同时保存场景信息
            scene_info = []
            
            for i, scene in enumerate(scenes):
                frame_num = scene[0].frame_num # 获取场景的开始帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num) # 设置视频读取的帧数
                ret, frame = cap.read() # 读取帧
                
                if ret:
                    # 添加时间戳信息
                    timestamp = frame_num / self.fps
                    output_path = output_dir / f"keyframe_{i:04d}_time_{timestamp:.2f}s.jpg"
                    cv2.imwrite(str(output_path), frame)
                    keyframe_paths.append(output_path)
                    
                    scene_info.append({
                        'frame_num': frame_num,
                        'timestamp': timestamp,
                        'duration': scene[1].get_seconds() - scene[0].get_seconds()
                    })
                    
                    print(f"保存关键帧 {i+1}/{len(scenes)}, 时间戳: {timestamp:.2f}s")
            
            # 保存场景信息
            import json
            with open(output_dir / 'scene_info.json', 'w') as f:
                json.dump(scene_info, f, indent=4)
                
            cap.release()
            return keyframe_paths, scene_info
        except Exception as e:
            print(f"保存关键帧时出错: {e}")
            return [], []

    def __del__(self):
        try:
           
            if hasattr(self, 'cap'):
                self.cap.release()
        except Exception as e:
            print(f"Error releasing video source: {e}")

    def extract_by_content(self):
        """
        优化的内容检测方法
        """
        try:
            print(f"\n使用内容检测:")
            print(f"- 检测阈值: {self.config['content_threshold']}")
            
            detector = ContentDetector(
                threshold=self.config['content_threshold'],
                min_scene_len=self.config['min_scene_len'],
                luma_only=self.config['luma_only']
            )
            
            scenes = detect(str(self.video_path), detector)
            if not scenes:
                print(f"没有检测到场景, 尝试降低阈值")
                self.config['content_threshold'] -= 1
                return self.extract_by_content()
            keyframes, scene_info = self._save_keyframes(scenes, "content")
            
            print(f"内容检测完成:")
            print(f"- 检测到 {len(scenes)} 个场景")
            if scene_info:
                print(f"- 平均场景时长：{np.mean([info['duration'] for info in scene_info]):.2f}秒")
            return keyframes, scene_info
        except Exception as e:
            print(f"内容检测失败: {e}")
            return [], []

    def extract_by_adaptive(self):
        """
        优化的自适应检测方法
        """
        try:
            print(f"\n使用自适应检测:")
            print(f"- 自适应阈值: {self.config['adaptive_threshold']}")
            
            detector = AdaptiveDetector(
                min_scene_len=self.config['min_scene_len'],
                adaptive_threshold=self.config['adaptive_threshold'],
                luma_only=self.config['luma_only'],
                window_width=2,
                min_content_val=15,
            )
            
            scenes = detect(str(self.video_path), detector)
            if not scenes:
                print(f"没有检测到场景, 尝试降低阈值")
                detector = AdaptiveDetector(
                    min_scene_len=self.config['min_scene_len'],
                    adaptive_threshold=1,
                    luma_only=self.config['luma_only'],
                    window_width=2,
                    min_content_val=15,
                )
                scenes = detect(str(self.video_path), detector)
            if not scenes:
                print(f"没有检测到场景, 建议尝试其他检测方法")
                return [], []
            keyframes, scene_info = self._save_keyframes(scenes, "adaptive")
            
            # 计算统计信息
            durations = [info['duration'] for info in scene_info]
            print(f"自适应检测完成:")
            print(f"- 检测到 {len(scenes)} 个场景")
            print(f"- 最短场景: {min(durations):.2f}秒")
            print(f"- 最长场景: {max(durations):.2f}秒")
            print(f"- 平均场景时长: {np.mean(durations):.2f}秒")
            
            return keyframes, scene_info
        except Exception as e:
            print(f"自适应检测失败: {e}")
            return [], []

    def extract_by_interval(self, interval=0.2):
        """基于间隔提取关键帧"""
        try:
            if not hasattr(self, 'fps'):
                self._init_video_info()
            if interval <= 0:
                raise ValueError(f"Invalid interval: {interval}")
            if interval > self.fps:
                raise ValueError(f"Interval is greater than the video's fps: {interval} > {self.fps}")
            print(f"fixed interval extract: {interval}s")
            print(f"视频信息:")
            print(f"- 总时长: {self.duration:.2f}秒")
            print(f"- 帧率: {self.fps}")
            print(f"- 总帧数: {self.frame_count}")
            print(f"- 提取间隔: {interval}秒")
            # 计算提取间隔的帧数
            frame_interval = int(self.fps * interval)
            # 理论应提取帧数
            total_frame = int(self.frame_count / frame_interval)
            output_dir = self.output_dir / "interval"
            output_dir.mkdir(exist_ok=True)
            keyframe_paths = []

            cap = cv2.VideoCapture(str(self.video_path))
            saved_count = 0
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 如果当前帧数是提取间隔的整数倍，则提取关键帧
                if frame_count % frame_interval == 0:
                    # 计算当前帧的时间戳    
                    timestamp = frame_count / self.fps
                    # 保存关键帧
                    output_path = output_dir / f"keyframe_{timestamp:.2f}s.jpg"
                    cv2.imwrite(str(output_path), frame)
                    keyframe_paths.append(output_path)
                    saved_count += 1
                    print(f"Saved keyframe: {saved_count} , current frame: {frame_count}")
                frame_count += 1
            cap.release()
            print(f"实际提取帧数: {saved_count}")
            print(f"理论应提取帧数: {total_frame}")
            # 允许误差1帧
            if abs(saved_count - total_frame)>1:
                print(f"实际提取帧数与理论应提取帧数不一致, 请检查视频是否被剪辑过")
                print(f"可能原因：")
                print(f"1. 视频帧率不均匀") 
                print(f"2. 视频末尾存在不完整帧")
                print(f"3. 舍入误差导致")
            return keyframe_paths
        except Exception as e:
            print(f"固定间隔提取失败: {e}")
            print(f"错误详情: {str(e)}")
            return []
        
    def evaluate_results(self, results):
        """
        评估不同方法的结果
        """
        try:
            print("\n结果评估:")
            
            # 遍历results中的每个方法，获取关键帧和场景信息
            # results是一个字典，key是方法名，value是一个元组，包含关键帧和场景信息 
            for method_name, result in results.items():
                try:
                    if not result:
                        print(f"{method_name}方法没有提取到关键帧")
                        continue
                    if isinstance(result, tuple) and len(result) == 2:
                        frames, info = result
                    elif isinstance(result, list):
                        frames,info = result,None
                    else:
                        print(f"\n{method_name}方法结果格式不正确")
                        continue
                    print(f"\n{method_name}方法评估:")
                    print(f"- 关键帧数量: {len(frames)}")

                    if info and isinstance(info, list):
                        try:
                            durations = [i['duration'] for i in info]
                            print(f"- 平均场景时长: {np.mean(durations):.2f}秒")
                            print(f"- 场景时长方差: {np.var(durations):.2f}")
                        except (KeyError, TypeError) as e:
                            print(f"无法计算场景时长: {str(e)}")
                    if hasattr(self, 'duration') and self.duration>0:
                        density = len(frames) / self.duration
                        print(f"- 检测密度: {density:.2f}帧/秒")
                except Exception as e:
                    print(f"评估{method_name}方法时出错: {str(e)}")
                    continue
        except Exception as e:
            print(f"结果评估失败: {str(e)}")

    def extract_all_methods(self):
        """
        提取所有方法的关键帧并评估
        """
        try:
            print("开始提取关键帧...")
            results = {
            "content": self._ensure_tuple(self.extract_by_content()),
            "adaptive": self._ensure_tuple(self.extract_by_adaptive()),
            "interval": self._ensure_tuple(self.extract_by_interval())
        }
        
            self.evaluate_results(results)
            return results
        except Exception as e:
            print(f"提取关键帧失败: {str(e)}")
            return {}
    def _ensure_tuple(self, result):
        """
        确保结果是一个元组
        Args:
            result: 输入结果，可能是列表或元组
        Returns:
            tuple: (frames, info) 格式的元组
        """
        return_value = ([],None)

        if isinstance(result, list):
            return_value = (result,None)
        elif isinstance(result, tuple) and len(result) == 2:
            return_value = result
        return return_value
    
@staticmethod
def print_statistics(results):
    print(f"\n ------------------------")
    print(f" 关键帧提取结果统计")
    print(f" ------------------------")
    for method_name, result in results.items():
        frames, _ = result
        frame_count = len(frames) if frames else 0
        if method_name =='interval':
            print(f"{method_name}方法提取了{frame_count}个关键帧")
        else:
            print(f"{method_name}方法提取了{frame_count}个关键帧")
        save_dir = Path(OUTPUT_DIR) / method_name
        if save_dir.exists():
            print(f"关键帧保存在: {save_dir}")
    total_frames = sum(len(frames) for frames in results.values())
    print(f"总共提取了{total_frames}个关键帧")
   
def extract_keyframes():
    try:    
        print("初始化关键帧提取器。。。。")
        extractor = KeyFrameExtractor(video_path=SOURCE_VIDEO, output_dir=OUTPUT_DIR)
        print("开始提取关键帧。。。")
        results = extractor.extract_all_methods()
        print(f"\n视频信息:")
        print(f"时长: {extractor.duration:.2f}秒")
        print(f"帧率: {extractor.fps}")
        print(f"总帧数: {extractor.frame_count}")
        print("结果统计：")
        print_statistics(results)
    except Exception as e:
        print(f"Error extracting keyframes: {e}")

if __name__ == "__main__":
    extract_keyframes()
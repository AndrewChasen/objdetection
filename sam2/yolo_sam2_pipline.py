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
OUTPUT_DIR = BASE_DIR / 'keyframes'
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
        """
        self.config = config or {
            "content_threshold": 2.0,
            "min_scene_len": 2,  
            "interval": 0.2,
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
        保存关键帧
        """
        try:
            keyframe_paths = []
            output_dir = self.output_dir / method_name
            output_dir.mkdir(exist_ok=True)
            
            # 使用cv2库来读取视频
            cap = cv2.VideoCapture(str(self.video_path))
            fram_numbers = [scene[0].frame_num for scene in scenes]
            for i,fram_number in enumerate(fram_numbers):
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, fram_number)
                ret, frame = cap.read()
                if ret:
                    output_path = output_dir / f"keyframe_{i:04d}.jpg" 
                    cv2.imwrite(output_path, frame)
                    keyframe_paths.append(output_path)
                    print(f"Saved keyframe {i} to {output_path}")
            cap.release()
            return keyframe_paths
        except Exception as e:
            print(f"Error saving keyframes: {e}")
            return []

    def __del__(self):
        try:
           
            if hasattr(self, 'cap'):
                self.cap.release()
        except Exception as e:
            print(f"Error releasing video source: {e}")

    def extract_by_content(self, threshold=2.0):
        """
        基于内容检测提取关键帧
        """
        # 使用scenedetect库来检测视频中的关键帧
        # 使用ContentDetector来检测视频中的关键帧
        # 整个过程就是封装成了一个api函数，直接调用这个函数即可就可以找到场景scenes了
        try:
            video_path = str(self.video_path)
            scenes = detect(video_path, ContentDetector(threshold=threshold))
            return self._save_keyframes(scenes, "content")
        except Exception as e:
            print(f"Error detecting scenes: {e}")
            return []        
        
    
    def extract_by_adaptive(self, min_scene_len=2):
        """
        基于自适应检测提取关键帧
        """
        try:
            print(f"\n使用自适应检测:")
            print(f"- 最小场景长度: {min_scene_len}帧")
            print(f"- 视频帧率: {self.fps}")
            print(f"- 视频总时长: {self.duration:.2f}秒")
            video_path = str(self.video_path)
            detector = AdaptiveDetector(
                min_scene_len=min_scene_len,  # 最小场景长度
            )
            scenes = detect(video_path, detector, show_progress=True)
            keyframes = self._save_keyframes(scenes, "adaptive")
            print(f"自适应检测完成: 找到 {len(keyframes)} 个场景")
            return keyframes   
        except Exception as e:
            print(f"Error detecting scenes: {e}")
            return []
    
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

            frame_interval = int(self.fps / interval)
            expected_num_frames = int(self.frame_count / frame_interval)
            output_dir = self.output_dir / "interval"
            output_dir.mkdir(exist_ok=True)
            keyframe_paths = []

            cap = cv2.VideoCapture(str(self.video_path))
            saved_count = 0
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    output_path = output_dir / f"keyframe_{frame_count:04d}.jpg"
                    cv2.imwrite(str(output_path), frame)
                    keyframe_paths.append(output_path)
                    saved_count += 1
                    print(f"Saved keyframe: {saved_count} , current frame: {frame_count}")
                frame_count += 1
            cap.release()
            print(f"实际提取帧数: {saved_count}")
            return keyframe_paths
        except Exception as e:
            print(f"固定间隔提取失败: {e}")
            print(f"错误详情: {str(e)}")
            return []
        
    def extract_all_methods(self):
        """提取所有方法的关键帧"""
        print("开始提取关键帧。。。")
        results = {}
        methods = {
            "content": self.extract_by_content(),
            "adaptive": self.extract_by_adaptive(),
            "interval": self.extract_by_interval(),
            # "difference": self.extract_by_difference()
        }

        for method, frames in methods.items():
            print(f"\n{method}方法:")
            print(f"- 提取了 {len(frames)} 个关键帧")
            print(f"- 保存位置: {self.output_dir}/{method}/")
        return results

    # def extract_by_difference(self, threshold=30):
    #     """基于图像差异提取关键帧"""
    #     keyframe_paths = []
    #     output_dir = self.output_dir / "difference"
    #     output_dir.mkdir(exist_ok=True)
        
    #     cap = cv2.VideoCapture(str(self.video_path))
    #     prev_frame = None
    #     frame_count = 0
        
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
                
    #         if prev_frame is not None:
    #             # 计算帧间差异
    #             diff = cv2.absdiff(frame, prev_frame)
    #             mean_diff = np.mean(diff)
                
    #             if mean_diff > threshold:
    #                 output_path = output_dir / f"keyframe_{frame_count:04d}.jpg"
    #                 cv2.imwrite(str(output_path), frame)
    #                 keyframe_paths.append(output_path)
                    
    #         prev_frame = frame.copy()
    #         frame_count += 1
            
    #     cap.release()
    #     return keyframe_paths

def extract_keyframes():
    try:    
        print("初始化关键帧提取器。。。。")
        extractor = KeyFrameExtractor(video_path=SOURCE_VIDEO, output_dir=OUTPUT_DIR,
                                      config={
                                          "content_threshold": 2.0,
                                          "min_scene_len": 2,  
                                          "interval": 0.2,
                                      })
        print("开始提取关键帧。。。")
        all_frames = extractor.extract_all_methods()
        print(f"\n视频信息:")
        print(f"时长: {extractor.duration:.2f}秒")
        print(f"帧率: {extractor.fps}")
        print(f"总帧数: {extractor.frame_count}")
        for method, frames in all_frames.items():
            print(f"{method} method has {len(frames)} keyframes")
            print(f"关键帧保存在: {OUTPUT_DIR}/{method}/")
        print(f"所有关键帧提取完成。。。")
        print("-"*50)
        print("结果统计：")
        for method, frames in all_frames.items():
            print(f"{method}方法提取了{len(frames)}个关键帧")
    except Exception as e:
        print(f"Error extracting keyframes: {e}")

if __name__ == "__main__":
    extract_keyframes()
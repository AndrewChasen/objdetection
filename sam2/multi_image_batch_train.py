import numpy as np
import torch
import cv2
import os

from torch.onnx.symbolic_opset11 import hstack
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from matplotlib import pyplot as plt

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


def prepare_data(data_dir):
    data = [] # list of files in the dataset
    try:
        # 构建图像目录路径
        image_dir = os.path.join(data_dir, "Simple/Train/Image/")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        
        # 获取目录中的所有图像
        image_files = os.listdir(image_dir)
        total_files = len(image_files)

        #list the dir and append them into the list with image and annotation.
        for ff, name in enumerate(image_files):
            # 构建图像和注释的完整路径
            image_path = os.path.join(data_dir, "Simple/Train/Image", name)
            annotation_path = os.path.join(data_dir, "Simple/Train/Instance", f"{name[:-4]}.png")

            # 验证文件是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}")
                continue
            if not os.path.exists(annotation_path):
                print(f"警告: 注释文件不存在: {annotation_path}")
                continue
            
            # 遍历数据目录中的所有图像文件
            # ff: 当前文件的索引
            # name: 当前文件的名称
            # 将图像文件路径和对应的注释文件路径添加到数据列表中
            # 图像文件路径: data_dir + "Simple/Train/Image/" + name
            # 注释文件路径: data_dir + "Simple/Train/Instance/" + name[:-4] + ".png"
            # 其中 name[:-4] 去掉了文件名的后缀（假设文件名后缀为4个字符，如 ".jpg"）
            # 添加到数据列表
            data.append({
                "image": image_path,
                "annotation": annotation_path
            })
            # 打印进度
            print(f"Processing {ff+1}/{total_files} files")

        if len(data) == 0:
                raise ValueError("没有找到有效的数据")
        print(f"成功加载 {len(data)} 个数据样本")
        return data
    except Exception as e:
        print(f"数据准备错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

#define a read_single function
def read_single(data): # read the random image and its annotation from the dataset
    try:
        # select image
        # 这行代码的意思是从数据列表中随机选择一个条目
        # np.random.randint(len(data)) 生成一个在 0 和 len(data) 之间的随机整数
        # data[np.random.randint(len(data))] 使用这个随机整数作为索引，从数据列表中选择一个条目
        # 这个条目包含图像文件路径和对应的注释文件路径
        # 选择的条目被赋值给变量 ent
        ent = data[np.random.randint(len(data))] # random function
        # 读取图像文件并将其转换为RGB格式
        # ent["image"] 是图像文件的路径
        # cv2.imread(ent["image"]) 读取图像文件
        # [...,::-1] 将图像从BGR格式转换为RGB格式
        # two differnt images on is the origin images one is the annotation
        Img = cv2.imread(ent["image"])[...,::-1] # read image
        ann_map = cv2.imread(ent["annotation"]) # read annotation
        print("原始图像形状:", Img.shape)
        print("原始注释图像形状:", ann_map.shape)
    
        #resize image
        # 这段代码的含义是计算图像的缩放比例，以确保图像的宽度和高度都不超过1024像素。
        # 具体来说，它计算了1024与图像宽度的比值（1024 / Img.shape[1]）和1024与图像高度的比值（1024 / Img.shape[0]），
        # 然后取这两个比值中的最小值作为缩放比例r。
        # 这样可以确保图像在缩放后，宽度和高度都不会超过1024像素，同时保持图像的宽高比不变。
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        # 使用最近邻插值方法调整注释图像的大小
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
        # 从高度上来补充
        if Img.shape[0]<1024:
            Img = np.concatenate([Img, np.zeros([1024 - Img.shape[0], Img.shape[1],3], dtype=np.uint8)], axis=0)
            ann_map = np.concatenate([ann_map,np.zeros([1024- ann_map.shape[0], ann_map.shape[1],3], dtype=np.uint8)],axis=0)
        # 从宽度上来补充
        if Img.shape[1]<1024:
            Img = np.concatenate([Img, np.zeros([Img.shape[0], 1024-Img.shape[1],3], dtype=np.uint8)], axis=1)
            ann_map = np.concatenate([ann_map, np.zeros([ann_map.shape[0], 1024- ann_map.shape[1], 3],dtype=np.uint8)], axis=1)
        #merge vessels and materials annotations
        # 这段代码的意思是将注释图像中的材料和容器注释合并到一个图像中
        # 首先，提取材料注释图像（mat_map）和容器注释图像（ves_map）
        # mat_map 是注释图像的第一个通道，表示材料注释
        # ves_map 是注释图像的第三个通道，表示容器注释
        # 然后，将材料注释图像中值为0的像素替换为容器注释图像中值为0的像素，并将其值加上材料注释图像的最大值加1
        # 最后，将材料注释图像、一个全零的图像和容器注释图像堆叠在一起，形成一个新的注释图像
        mat_map = ann_map[:,:,0] # material annotation map
        ves_map = ann_map[:,:,2] # vessels annotation map
    
        # mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1) # merge maps
        mask = mat_map == 0
        ves_values = ves_map[mask]
        mat_map[mask] = ves_values * (mat_map.max() + 1 if mat_map.max() > 0 else 1)
    
        inds= np.unique(mat_map)[1:]  # ignore the 0 position

        if inds.__len__()>0:
            ind = inds[np.random.randint(inds.__len__())]
        else:
            return read_single(data)
        
        mask =(mat_map ==ind).astype(np.uint8) #mask 
        coords = np.argwhere(mask >0) # 会返回所有非零位置的坐标 行和列，所有有yx[1], yx[0]的说法
        yx = np.array(coords[np.random.randint(len(coords))])

        return Img, mask, [[yx[1], yx[0]]]
    
    except Exception as e:
            print(f"读取样本错误: {str(e)}")
            return None, None, None

def read_batch(data, batch_size=4):
    limage = []
    lmask = []
    linput_point = []
    try: 
        for  i in range(batch_size):
            image, mask, input_point = read_single(data)
            if image is None or mask is None or input_point is None:
                print(f"跳过无效样本 {i+1}")
                continue
            limage.append(image)
            lmask.append(mask)
            linput_point.append(input_point)
        if len(limage) == 0:
            print("没有有效样本")
            return None, None, None, None
        # 确保返回的批次大小正确
        actual_batch_size = len(limage)
            # 会创建一个4行1列的数组，所有元素都是1 np.ones([actual_batch_size,1]
        return limage, np.array(lmask), np.array(linput_point), np.ones([actual_batch_size,1])
    except Exception as e:
        print(f"批量读取错误: {str(e)}")
        return None, None, None, None


def initialize_model(checkpoint_path, model_cfg, device):
    try:
        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
            
        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建完整的配置文件路径
        full_config_path = os.path.join(current_dir, "sam2", model_cfg)
        if not os.path.exists(full_config_path):
            raise FileNotFoundError(f"配置文件不存在: {full_config_path}")

        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        if sam2_model is None:
            raise RuntimeError("模型构建失败")
        predictor = SAM2ImagePredictor(sam2_model)
        # 确保模型在正确的设备上
        sam2_model = sam2_model.to(device)
        #set training parameters
        predictor.model.sam_mask_decoder.train(True) #enable traing of mask decoder
        predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
        predictor.model.image_encoder.train(True)
        print(f"模型已成功加载到设备: {device}")
        return sam2_model, predictor
    except Exception as e:
        print(f"模型初始化错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None

def setup_optimizer(predictor):
    try:
        # 检查模型参数
        if not list(predictor.model.parameters()):
            raise ValueError("模型没有可训练的参数")
        # 这段代码的含义是创建一个优化器，用于训练SAM2模型的参数。
        # 具体来说，它使用AdamW优化器，并设置学习率为1e-5，权重衰减为4e-5。
        # optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
        return torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    except Exception as e:
        print(f"优化器设置错误: {str(e)}")
        return None

def train_sam2(
        data_dir:str, 
        checkpoint_path:str,
        model_cfg:str,
        max_epochs:int=100,
        patience:int=10):
    try:
        data = prepare_data(data_dir)
        if data is None:
            return

        device = setup_device()
        model, predictor = initialize_model(checkpoint_path,model_cfg,device)
        if model is None or predictor is None:
            return
        optimizer = setup_optimizer(predictor)
        if optimizer is None:
            return
        history = train_loop(data, predictor,optimizer,model,max_epochs=max_epochs, 
                           patience=patience)

        return history
    except Exception as e:
        print(f"训练过程错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

    
    # scaler = torch.amp.GradScaler(device=device.type)  

def train_loop(data, predictor,optimizer,model,max_epochs=100, patience=10):
    best_iou = 0
    no_improve = 0
    history = {'iou':[], 'loss':[]}
    device=next(model.parameters()).device 
    mean_iou = 0
    try:
        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch+1}/{max_epochs}")
            epoch_ious = []
            epoch_losses = []

            for itr in range(30):
                            
                image, mask, input_point, input_label = read_batch(data, batch_size=4) # load data batch
                if image is None or mask is None:
                        print("跳过无效数据批次")
                        continue      
                if mask.shape[0] == 0: 
                    print("跳过空掩码批次")
                    continue # ignore empty batches
                    # 这段代码的含义是将图像、掩码、输入点和输入标签传递给预测器，并设置图像。
                    # 具体来说，它首先调用predictor.set_image(image)方法，将图像传递给预测器。
                    # 然后，它调用predictor._prep_prompts方法，准备输入点和输入标签的提示。
                    # 该方法返回掩码输入、未归一化的坐标、标签和未归一化的框。
                    # 最后，这些返回值可以用于进一步的处理或训练。
                with torch.amp.autocast(device_type=device.type):
                    predictor.set_image_batch(image) # apply sam image encoder to the image

                    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, 
                                                                                            box=None, mask_logits=None, 
                                                                                            normalize_coords=True)
                    # 这段代码的含义是将稀疏嵌入和稠密嵌入传递给SAM2模型的提示编码器。
                    # 具体来说，它调用predictor.model.sam_prompt_encoder方法，并传递点、框和掩码作为参数。
                    # 该方法返回稀疏嵌入和稠密嵌入，这些嵌入可以用于进一步的处理或训练。
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), 
                                                                                            boxes=None, 
                                                                                            masks=None,)

                    # high resolution pictures
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                    image_embeddings = predictor._features["image_embed"]
                    # 图片特征跟我们的提示点的特征是否相等,
                    if image_embeddings.shape[0] !=sparse_embeddings.shape[0]:
                        image_embeddings = image_embeddings.repeat(sparse_embeddings.shape[0],1,1,1)
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=image_embeddings,
                                                                                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                                                                                    sparse_prompt_embeddings=sparse_embeddings,
                                                                                    dense_prompt_embeddings=dense_embeddings,
                                                                                    multimask_output=True,
                                                                                    repeat_image=False,
                                                                                    high_res_features=high_res_features,)
                    
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, 
                                                                        predictor._orig_hw[-1])# Upscale the masks to the original image resolution

                    # 这段代码的含义是计算分割损失和交并比（IoU）损失，并将它们结合起来得到最终的损失值。
                    # 首先，使用预测的掩码和真实掩码计算分割损失。分割损失是通过交叉熵损失计算的，
                    # 其中预测的掩码通过sigmoid函数进行归一化。然后，计算交并比（IoU）损失，
                    # 交并比是通过计算预测掩码和真实掩码的交集和并集来计算的。
                    # 最后，将分割损失和交并比损失结合起来，得到最终的损失值。
                    # 其中，分割损失和交并比损失的权重分别为1和0.05。
                    prd_mask = torch.sigmoid(prd_masks[:,0]) # 预测出来的mask
                    gt_mask = torch.tensor(mask.astype(np.float32)).to(device=device) #我们自己的mask

                    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1- gt_mask) * torch.log((1- prd_mask) + 0.00001)).mean()

                    #score loss calculation iou
                    inter= (gt_mask* (prd_mask>0.5)).sum(1).sum(1) # calculate from the height and width.
                    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    loss = seg_loss + score_loss*0.05

                    #apply back propogation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    current_iou = iou.mean().item()
                    epoch_ious.append(current_iou)
                    epoch_losses.append(loss.item())

                    if itr % 3 ==0: 
                        torch.save(predictor.model.state_dict(), "model.torch") # save model

                    if itr % 3 ==0:          
                        mean_iou = mean_iou * 0.99 +0.01 * np.mean(iou.cpu().detach().numpy())
                        print(f"Iteration {itr}, Mean IoU: {mean_iou}")
                    print(f"running....{itr}")
            avg_iou = np.mean(epoch_ious)
            avg_loss = np.mean(epoch_losses)
            history['iou'].append(avg_iou)
            history['loss'].append(avg_loss)
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average IoU: {avg_iou:.4f}")
            print(f"Average Loss: {avg_loss:.4f}")

            if avg_iou> best_iou:
                best_iou = avg_iou
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': predictor.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou,
                    'loss': avg_loss
                }, "best_model.pth")
                print(f"保存新的最佳模型,IoU: {best_iou:.4f}")
            else:
                no_improve += 1
                print(f"模型性能未提升，已经 {no_improve} 轮")
            if no_improve >=patience:
                print(f"模型性能已经 {patience} 轮未提升，停止训练！")
                break
        plot_training_history(history)
    except Exception as e:
        print(f"训练出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # 绘制IoU曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['iou'])
    plt.title('IoU over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    
    # 绘制Loss曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 项目根目录
    print(f"项目根目录: {project_root}")
    try:
        import hydra
        from hydra import initialize, compose
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        os.chdir(current_dir)
        initialize(config_path="sam2", version_base=None)

        config = {
        "data_dir": os.path.join(current_dir, "assets/LabPicsV1/"),
        "checkpoint_path": os.path.join(current_dir, "checkpoints/sam2.1_hiera_tiny.pt"),
        "model_cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "max_epochs": 100,
        "patience": 10 
        }
        
        
        # 检查文件是否存在
        full_config_path = os.path.join(current_dir, "sam2", config["model_cfg"])
        if not os.path.exists(full_config_path):
            raise FileNotFoundError(f"配置文件不存在: {full_config_path}")
        
        print("开始训练...")
        history = train_sam2(**config)
        print("训练完成！")
    except Exception as e:
        print(f"发生的错误:{str(e)}")   
        import traceback
        print(traceback.format_exc())
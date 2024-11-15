import numpy as np
import torch
import cv2
import os
import sys


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = 'cpu' if torch.backends.mps.is_available() else 'mps'

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"Using device: {device}")

device = torch.device(device=device)

data_dir = "./assets/LabPicsV1/" # path to the dataset
data = [] # list of files in the dataset

for ff, name in enumerate(os.listdir(data_dir + "Simple/Train/Image/")):
    # 遍历数据目录中的所有图像文件
    # ff: 当前文件的索引
    # name: 当前文件的名称
    # 将图像文件路径和对应的注释文件路径添加到数据列表中
    # 图像文件路径: data_dir + "Simple/Train/Image/" + name
    # 注释文件路径: data_dir + "Simple/Train/Instance/" + name[:-4] + ".png"
    # 其中 name[:-4] 去掉了文件名的后缀（假设文件名后缀为4个字符，如 ".jpg"）
    data.append({"image":data_dir+"Simple/Train/Image/"+name, "annotation":data_dir+"Simple/Train/Instance/"+name[:-4]+".png"})
    print(f"Processing {ff} of {len(os.listdir(data_dir + 'Simple/Train/Image/'))}")

def read_batch(data): # read the random image and its annotation from the dataset
    # select image
    # 这行代码的意思是从数据列表中随机选择一个条目
    # np.random.randint(len(data)) 生成一个在 0 和 len(data) 之间的随机整数
    # data[np.random.randint(len(data))] 使用这个随机整数作为索引，从数据列表中选择一个条目
    # 这个条目包含图像文件路径和对应的注释文件路径
    # 选择的条目被赋值给变量 ent
    ent = data[np.random.randint(len(data))]
    # 读取图像文件并将其转换为RGB格式
    # ent["image"] 是图像文件的路径
    # cv2.imread(ent["image"]) 读取图像文件
    # [...,::-1] 将图像从BGR格式转换为RGB格式
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
    
    #merge vessels and materials annotations
    # 这段代码的意思是将注释图像中的材料和容器注释合并到一个图像中
    # 首先，提取材料注释图像（mat_map）和容器注释图像（ves_map）
    # mat_map 是注释图像的第一个通道，表示材料注释
    # ves_map 是注释图像的第三个通道，表示容器注释
    # 然后，将材料注释图像中值为0的像素替换为容器注释图像中值为0的像素，并将其值加上材料注释图像的最大值加1
    # 最后，将材料注释图像、一个全零的图像和容器注释图像堆叠在一起，形成一个新的注释图像
    mat_map = ann_map[:,:,0] # material annotation map
    ves_map = ann_map[:,:,2] # vessels annotation map

    print("mat_map shape:", mat_map.shape)
    print("ves_map shape:", ves_map.shape)
    print("Number of zeros in mat_map:", (mat_map==0).sum())
    print("Number of zeros in ves_map:", (ves_map==0).sum())
    print("mat_map 唯一值:", np.unique(mat_map))
    print("ves_map 唯一值:", np.unique(ves_map))
  
    # mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1) # merge maps
    mask = mat_map == 0
    ves_values = ves_map[mask]
    mat_map[mask] = ves_values * (mat_map.max() + 1 if mat_map.max() > 0 else 1)
   
   # get binary masks and points
    inds = np.unique(mat_map[1:]) # load all indices in a line.
    points = []
    masks = []
    for ind in inds:
        mask = (mat_map ==ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask >0) # returns the coordinates of the pixels that are greater than 0
        yx = np.array(coords[np.random.randint(len(coords))]) # randomly select a point from the coordinates
        points.append([[yx[1],yx[0]]]) # append the coordinates of the selected point
    return Img, np.array(masks), np.array(points), np.ones([len(masks),1])

# load the model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

#set training parameters
predictor.model.sam_mask_decoder.train(True) #enable traing of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

# 这段代码的含义是创建一个优化器，用于训练SAM2模型的参数。
# 具体来说，它使用AdamW优化器，并设置学习率为1e-5，权重衰减为4e-5。
# optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)

optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
# scaler = torch.amp.GradScaler(device=device.type)  
mean_iou = 0   
try:
    for itr in range(30):
        print(f"running....{itr}")
        with torch.amp.autocast(device_type=device.type):
            image, mask, input_point, input_label = read_batch(data) # load data batch
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
            predictor.set_image(image) # apply sam image encoder to the image

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, 
                                                                                    box=None, mask_logits=None, 
                                                                                    normalize_coords=True)
            # 这段代码的含义是将稀疏嵌入和稠密嵌入传递给SAM2模型的提示编码器。
            # 具体来说，它调用predictor.model.sam_prompt_encoder方法，并传递点、框和掩码作为参数。
            # 该方法返回稀疏嵌入和稠密嵌入，这些嵌入可以用于进一步的处理或训练。

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), 
                                                                                    boxes=None, 
                                                                                    masks=None,)

            
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            image_embeddings = predictor._features["image_embed"]
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
            prd_mask = torch.sigmoid(prd_masks[:,0])
            gt_mask = torch.tensor(mask.astype(np.float32)).to(device=device)

            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1- gt_mask) * torch.log((1- prd_mask) + 0.00001)).mean()

            #score loss calculation iou
            inter= (gt_mask* (prd_mask>0.5)).sum(1).sum(1) # calculate from the height and width.
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        

            loss = seg_loss + score_loss*0.05

            #apply back propogation

            predictor.model.zero_grad()
            # scaler.scale(loss).backward() # backpropogate
            # scaler.step(optimizer)
            # scaler.update()
            predictor.model.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % 3 ==0: 
                torch.save(predictor.model.state_dict(), "model.torch") # save model

            if itr % 3 ==0:          
                mean_iou = mean_iou * 0.99 +0.01 * np.mean(iou.cpu().detach().numpy())
                print(f"Iteration {itr}, Mean IoU: {mean_iou}")
            print(f"running....{itr}")
except Exception as e:
    print(f"发生的错误:{str(e)}")
    import traceback
    print(traceback.format_exc())

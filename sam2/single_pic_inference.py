import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = 'cpu' if torch.backends.mps.is_available() else 'mps'
torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()


#load image
image_path = "test/sample_image.jpg"
mask_path = "test/sample_mask.png"

def read_image(image_path, mask_path):
    img = cv2.imread(image_path)[...,::-1]
    mask = cv2.imread(mask_path, 0)

    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    return img, mask

image, mask = read_image(image_path, mask_path)
print(f"read the image")
num_samples = 30

def get_points(mask, num_points):
    points =[]
    for i in range(num_points):
        coords = np.argwhere(mask >0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)
print(f"生成了 {num_samples} 个点，并将其存储在一个列表中")

input_points = get_points(mask, num_samples)

sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml"


sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

predictor.model.load_state_dict(torch.load("model.torch"))
print
with torch.no_grad():
    predictor.set_image(image)
    # 这段代码的作用是使用一个图像预测器（SAM2ImagePredictor）对输入图像进行分割预测。
    # 首先，代码从指定路径加载图像和掩码，并对它们进行预处理（调整大小）。
    # 然后，代码生成一组随机点，这些点位于掩码中非零像素的位置。
    # 接着，代码加载一个预训练的SAM2模型，并使用该模型创建一个图像预测器。
    # 预测器加载模型的状态字典，并将输入图像设置为预测器的图像。
    # 最后，代码使用预测器对输入点进行预测，生成分割掩码、分数和logits。
    masks, scores, logits = predictor.predict(
        point_coords = input_points,
        point_labels=np.ones(input_points.shape[0])[:,None], 
    )


masks = masks[:,0].astype(bool)

print(f"从大到小排列的scores值:")
shorted_masks = masks[np.argsort(scores[:,0])][::-1].astype(bool)

seg_map = np.zeros_like(shorted_masks[0],dtype=np.uint8)
occupancy_mask = np.zeros_like(shorted_masks[0],dtype=bool)

for i in range(shorted_masks.shape[0]):
    mask = shorted_masks[i]
    if (mask*occupancy_mask).sum() / mask.sum()>0.15: continue
    mask[occupancy_mask] = 0
    seg_map[mask]= i+1
    occupancy_mask[mask]= 1

height, width = seg_map.shape

rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1],3), dtype=np.uint8)

for id_class in range(1, seg_map.max()+1):
    rgb_image[seg_map ==id_class] = [np.random.randint(255),
                                    np.random.randint(255),
                                    np.random.randint(255)]

cv2.imwrite("annotation.png", rgb_image)
cv2.imwrite("mix.png",(rgb_image/2+image/2).astype(np.uint8))

# cv2.imshow("annotation",rgb_image)
cv2.imshow("mix",(rgb_image/2+image/2).astype(np.uint8))
# cv2.imshow("image",image)
print("图片mix完成,现实完成")
while True:
    key = cv2.waitKey(1) & 0xFF  # 使用位运算确保兼容性
    if key == 27:  # ESC键的ASCII码是27
        break
cv2.destroyAllWindows()
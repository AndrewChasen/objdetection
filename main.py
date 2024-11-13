import cv2
from ultralytics import YOLO
import numpy as np
# 该脚本使用OpenCV和YOLO模型进行对象检测
# 通过cv2.VideoCapture读取视频文件"dogs.mp4"
# YOLO模型来自ultralytics库
# 使用numpy进行数组操作

cap = cv2.VideoCapture("./objdetection/data/dogs.mp4")

model = YOLO("./objdetection/models/yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    results = model(frame, device="mps")
    print(results)
    results = results[0]
    print(results)

    bboxes = np.array(results.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(results.boxes.cls.cpu(), dtype="int")
    print(bboxes)
    print(classes)

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x,y), (x2,y2),(0,0,255),2)
        cv2.putText(frame, str(cls), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
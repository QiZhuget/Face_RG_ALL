import argparse
import cv2
import numpy as np
import os
import time
import dlib

# Dlib 5点特征点模型路径
PREDICTOR_PATH = "weights/shape_predictor_5_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def align_face(img, landmarks, desired_width=112):
    """人脸对齐：根据双眼位置做相似变换矫正"""
    right_eye = np.array([landmarks.part(0).x, landmarks.part(0).y], dtype=np.float32)
    left_eye = np.array([landmarks.part(2).x, landmarks.part(2).y], dtype=np.float32)

    desired_right = np.array([desired_width * 0.7, desired_width * 0.46], dtype=np.float32)
    desired_left = np.array([desired_width * 0.3, desired_width * 0.46], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(np.array([right_eye, left_eye]), np.array([desired_right, desired_left]))
    aligned = cv2.warpAffine(img, M, (desired_width, desired_width), flags=cv2.INTER_CUBIC)
    return aligned

# YOLO人脸检测 + dlib人脸对齐— 使用5点模型
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/yolov8n-face-lindevs.onnx', help='Weights path')
    parser.add_argument('--source', default='test_images/0016.jpg')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.75, help='Object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='Intersection union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output', default='output', help='Output directory for results')
    opt = parser.parse_args()

    # 加载 YOLOv8-Face ONNX 模型
    model = cv2.dnn.readNetFromONNX(opt.weights)

    # 读取图片
    img = cv2.imread(opt.source)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {opt.source}")
    height, width, _ = img.shape
    length = max((height, width))
    scale = length / opt.imgsz

    # 预处理：补成正方形 + 归一化
    blob = np.zeros((length, length, 3), np.uint8)
    blob[0:height, 0:width] = img
    blob = cv2.dnn.blobFromImage(blob, scalefactor=1 / 255, size=(opt.imgsz, opt.imgsz), swapRB=True)

    # 推理
    model.setInput(blob)
    outputs = model.forward()
    outputs = cv2.transpose(outputs[0])

    boxes = []
    scores = []
    for i in range(outputs.shape[0]):
        max_score = float(np.amax(outputs[i][4:]))
        if max_score >= opt.conf:
            boxes.append([
                (outputs[i][0] - 0.5 * outputs[i][2]) * scale,
                (outputs[i][1] - 0.5 * outputs[i][3]) * scale,
                outputs[i][2] * scale,
                outputs[i][3] * scale
            ])
            scores.append(max_score)

    # NMS 非极大值抑制
    results = cv2.dnn.NMSBoxes(boxes, scores, opt.conf, opt.iou)

    color = [0, 255, 0]
    face_count = 0

    # 遍历 YOLO 检测到的人脸
    for idx in results:
        if isinstance(idx, (list, np.ndarray)):
            idx = idx[0]

        x = round(boxes[idx][0])
        y = round(boxes[idx][1])
        w = round(boxes[idx][2])
        h = round(boxes[idx][3])

        # 转为 dlib 需要的 rectangle 格式
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)

        # 检测 5 点特征
        shape = predictor(img, dlib_rect)
        # renderFace(img, shape)

        # ===================== 人脸对齐 =====================
        aligned_face = align_face(img, shape)
        # ====================================================

        # 画框 + 置信度
        label = f'face {scores[idx]:.2f}'
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        face_count += 1

        # 保存对齐后的人脸
        os.makedirs(opt.output, exist_ok=True)
        input_filename = os.path.basename(opt.source)
        align_name = f"aligned_{face_count}_{input_filename}"
        align_path = os.path.join(opt.output, align_name)
        cv2.imwrite(align_path, aligned_face)

    # 保存原图检测结果
    os.makedirs(opt.output, exist_ok=True)
    input_filename = os.path.basename(opt.source)
    output_path = os.path.join(opt.output, f"detected_{input_filename}")
    cv2.imwrite(output_path, img)

    print(f"✅ 检测完成！")
    print(f"📁 检测结果保存到: {output_path}")
    print(f"📁 对齐人脸保存到: {opt.output}")
    print(f"👤 检测到 {face_count} 张人脸，已完成正脸矫正")
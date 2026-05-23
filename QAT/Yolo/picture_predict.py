import argparse
import cv2
import numpy as np
import os
import onnxruntime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='../../weights/yolov8n-face-lindevs-int8.onnx', help='Weights path')
    parser.add_argument('--source', default='../../test_images/0016.jpg')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.75, help='Object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--device', default='cpu', help='Execution provider: cpu, cuda, or tensorrt')
    parser.add_argument('--output', default='output', help='Output directory')
    opt = parser.parse_args()

    # ========== 1. 加载 ONNX Runtime 模型 ==========
    # 设置执行提供者（优先使用 CUDA 如果可用）
    providers = []
    if opt.device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif opt.device == 'tensorrt':
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    session = onnxruntime.InferenceSession(opt.weights, providers=providers)
    print(f"使用设备: {session.get_providers()}")

    # 获取模型输入输出信息
    input_name = session.get_inputs()[0].name
    print(f"模型输入名称: {input_name}, 形状: {session.get_inputs()[0].shape}")

    # ========== 2. 读取并预处理图像 ==========
    img = cv2.imread(opt.source)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {opt.source}")

    height, width, _ = img.shape
    length = max(height, width)
    scale = length / opt.imgsz

    # 创建正方形画布并填充原图
    canvas = np.zeros((length, length, 3), dtype=np.uint8)
    canvas[0:height, 0:width] = img

    # 转换为模型输入 blob (1,3,640,640) 归一化到 [0,1]
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, size=(opt.imgsz, opt.imgsz), swapRB=True)

    # ========== 3. 推理 ==========
    outputs = session.run(None, {input_name: blob})[0]  # 输出 shape: (1, 84, 8400)
    outputs = cv2.transpose(outputs[0])  # 转置为 (8400, 84)

    # ========== 4. 解析检测结果 ==========
    boxes = []
    scores = []
    for i in range(outputs.shape[0]):
        max_score = float(np.amax(outputs[i][4:]))
        if max_score >= opt.conf:
            # 转换坐标：cx, cy, w, h -> x, y, w, h
            x = (outputs[i][0] - 0.5 * outputs[i][2]) * scale
            y = (outputs[i][1] - 0.5 * outputs[i][3]) * scale
            w = outputs[i][2] * scale
            h = outputs[i][3] * scale
            boxes.append([x, y, w, h])
            scores.append(max_score)

    # ========== 5. NMS ==========
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, opt.conf, opt.iou)
        if indices is not None and len(indices) > 0:
            # 处理 indices 的不同返回格式
            if isinstance(indices[0], (list, np.ndarray)):
                indices = [idx[0] for idx in indices]
            else:
                indices = indices.flatten().tolist()
        else:
            indices = []
    else:
        indices = []

    # ========== 6. 绘制检测框 ==========
    color = [0, 255, 0]
    for idx in indices:
        x, y, w, h = map(int, boxes[idx])
        label = f'{scores[idx]:.2f}'
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ========== 7. 保存结果 ==========
    os.makedirs(opt.output, exist_ok=True)
    input_filename = os.path.basename(opt.source)
    output_filename = f"detected_{input_filename}"
    output_path = os.path.join(opt.output, output_filename)
    cv2.imwrite(output_path, img)

    print(f"检测结果已保存到: {output_path}")
    print(f"检测到 {len(indices)} 个人脸")
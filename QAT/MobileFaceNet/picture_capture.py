import cv2
import numpy as np
import os
import dlib

# Dlib 5点特征点模型路径
PREDICTOR_PATH = "../../weights/shape_predictor_5_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 保存目录（自动创建）
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

def align_face(img, landmarks, desired_width=112):
    """人脸对齐：根据双眼位置做相似变换矫正"""
    right_eye = np.array([landmarks.part(0).x, landmarks.part(0).y], dtype=np.float32)
    left_eye = np.array([landmarks.part(2).x, landmarks.part(2).y], dtype=np.float32)

    desired_right = np.array([desired_width * 0.7, desired_width * 0.46], dtype=np.float32)
    desired_left = np.array([desired_width * 0.3, desired_width * 0.46], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(np.array([right_eye, left_eye]), np.array([desired_right, desired_left]))
    aligned = cv2.warpAffine(img, M, (desired_width, desired_width), flags=cv2.INTER_CUBIC)
    return aligned

def detect_and_align_face(frame):
    """使用YOLO检测人脸 + dlib对齐"""
    # YOLO 人脸检测 ONNX 模型
    yoloface_model = cv2.dnn.readNetFromONNX("../../weights/yolov8n-face-lindevs.onnx")
    conf_thres = 0.75
    iou_thres = 0.7
    imgsz = 640

    height, width = frame.shape[:2]
    length = max(height, width)
    scale = length / imgsz

    # 预处理
    blob = np.zeros((length, length, 3), np.uint8)
    blob[0:height, 0:width] = frame
    blob = cv2.dnn.blobFromImage(blob, scalefactor=1/255, size=(imgsz, imgsz), swapRB=True)

    yoloface_model.setInput(blob)
    outputs = yoloface_model.forward()
    outputs = cv2.transpose(outputs[0])

    boxes = []
    scores = []
    for i in range(outputs.shape[0]):
        max_score = float(np.amax(outputs[i][4:]))
        if max_score >= conf_thres:
            boxes.append([
                (outputs[i][0] - 0.5 * outputs[i][2]) * scale,
                (outputs[i][1] - 0.5 * outputs[i][3]) * scale,
                outputs[i][2] * scale,
                outputs[i][3] * scale
            ])
            scores.append(max_score)

    # NMS
    results = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    aligned_faces = []

    if len(results) > 0:
        for idx in results.flatten():
            x = round(boxes[idx][0])
            y = round(boxes[idx][1])
            w = round(boxes[idx][2])
            h = round(boxes[idx][3])

            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            shape = predictor(frame, dlib_rect)
            aligned_face = align_face(frame, shape)
            aligned_faces.append(aligned_face)

    return aligned_faces

if __name__ == '__main__':
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        exit()

    print("=== 摄像头已启动 ===")
    print("👉 按【空格】拍照并进行人脸对齐")
    print("👉 按【ESC】退出程序")

    img_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取摄像头失败")
            break

        # 显示画面
        show_frame = frame.copy()
        cv2.putText(show_frame, "Press SPACE to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera - Face Align", show_frame)

        # 按键监听
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 退出
            break
        elif key == 32:  # 空格 拍照
            img_count += 1
            print(f"\n📸 已拍照 {img_count}")

            # 人脸检测 + 对齐
            aligned_faces = detect_and_align_face(frame)

            if len(aligned_faces) == 0:
                print("❌ 未检测到人脸")
            else:
                # 只保存对齐后的人脸（删除了原图保存）
                for i, face in enumerate(aligned_faces):
                    save_path = os.path.join(SAVE_DIR, f"face_{img_count}_{i+1}.jpg")
                    cv2.imwrite(save_path, face)
                    print(f"✅ 对齐人脸已保存：{save_path}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 程序已退出")
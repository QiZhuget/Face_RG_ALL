import argparse
import cv2
import dlib
import numpy as np
import os
import time

# YOLO人脸检测 + dlib人脸对齐 (自定义命名保存 + 自动退出)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', default='weights/yolov8n-face-lindevs.onnx', help='Yolo weights path')
    parser.add_argument('--dlib_shape_predictor', default='weights/shape_predictor_68_face_landmarks.dat',
                        help='dlib shape predictor path')
    parser.add_argument('--source', default='1', help='Video path or camera index (0, 1, 2...)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.75, help='Object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='Intersection over union (IoU) threshold for NMS')
    parser.add_argument('--output', default='face_database', help='Output directory for aligned faces')
    opt = parser.parse_args()

    # 加载 Yolo 模型
    yolo_model = cv2.dnn.readNetFromONNX(opt.yolo_weights)
    print(f"YOLO模型加载成功: {opt.yolo_weights}")

    # 加载 dlib 人脸检测器和形状预测器
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(opt.dlib_shape_predictor)
        print(f"dlib模型加载成功: {opt.dlib_shape_predictor}")
    except Exception as e:
        print(f"dlib模型加载失败: {e}")
        exit(1)

    # 打开视频源（直接使用摄像头0）
    cap = cv2.VideoCapture(0)
    source_name = f"camera_{0}"

    if not cap.isOpened():
        print(f"无法打开摄像头{0}")
        exit(1)

    # 设置摄像头参数（稳定画面）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    print(f"视频源: {source_name}")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps}")

    def detect_face(img, model, imgsz, conf_threshold, iou_threshold):
        """使用YOLO检测人脸（仅返回坐标，不绘制）"""
        height, width, _ = img.shape
        length = max((height, width))
        scale = length / imgsz

        # 创建正方形画布
        blob = np.zeros((length, length, 3), np.uint8)
        blob[0:height, 0:width] = img
        blob = cv2.dnn.blobFromImage(blob, scalefactor=1 / 255, size=(imgsz, imgsz), swapRB=True)

        # 推理
        model.setInput(blob)
        outputs = model.forward()
        outputs = cv2.transpose(outputs[0])

        # 解析结果
        boxes = []
        scores = []
        for i in range(outputs.shape[0]):
            max_score = float(np.amax(outputs[i][4:]))
            if max_score >= conf_threshold:
                boxes.append([
                    (outputs[i][0] - (0.5 * outputs[i][2])) * scale,
                    (outputs[i][1] - (0.5 * outputs[i][3])) * scale,
                    outputs[i][2] * scale,
                    outputs[i][3] * scale,
                ])
                scores.append(max_score)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        detections = []
        if indices is not None and len(indices) > 0:
            for idx in indices:
                idx = idx if isinstance(idx, (int, np.integer)) else idx[0]
                x, y, w, h = boxes[idx]
                score = scores[idx]
                detections.append([x, y, w, h, score])

        return detections

    # dlib人脸对齐
    def align_face(img, face_rect):
        """使用dlib进行人脸对齐，返回112x112标准化人脸"""
        # 转换为dlib的矩形格式
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        # 检测关键点
        shape = predictor(img, dlib_rect)

        # 获取关键点坐标
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        # 定义标准关键点位置
        LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
        RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

        # 计算眼睛中心点
        left_eye = landmarks[LEFT_EYE_INDICES].mean(axis=0)
        right_eye = landmarks[RIGHT_EYE_INDICES].mean(axis=0)

        # 计算眼睛连线的角度
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # 确保角度在合理范围内
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        # 计算眼睛中心
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # 计算仿射变换矩阵
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        # 应用变换
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # 裁剪对齐后的头像（包含更多头部区域）
        aligned_face = aligned[int(y):int(y + h), int(x):int(x + w)]

        # 调整为112x112大小（固定输出尺寸）
        aligned_face = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LANCZOS4)

        # 检查是否需要翻转（解决镜像问题）
        if left_eye[0] > right_eye[0]:
            aligned_face = cv2.flip(aligned_face, 1)

        return aligned_face


    # 创建输出目录（改为face_database）
    os.makedirs(opt.output, exist_ok=True)
    print(f"对齐后的人脸将保存到: {os.path.abspath(opt.output)}")

    # 处理视频帧
    processed_frames = 0
    start_time = time.time()
    save_count = 0

    print("\n开始实时检测并对齐人脸！")
    print("📷 画面已显示（无任何标注）")
    print("按 'q' 退出，按 's' 拍照并保存对齐人脸（112x112）")

    # 创建窗口并显示纯净画面
    cv2.namedWindow("Camera View (No Annotations)", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取摄像头画面，重试...")
            time.sleep(0.1)
            continue

        processed_frames += 1

        # 检测人脸（仅计算，不绘制任何框/文字）
        detections = detect_face(frame, yolo_model, opt.imgsz, opt.conf, opt.iou)

        # 显示纯净的摄像头画面（无任何标注）
        cv2.imshow("Camera View (No Annotations)", frame)

        # 按键控制（仅保留核心功能）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\n===== 开始保存人脸 =====")
            if len(detections) == 0:
                print("❌ 未检测到人脸，请调整位置后重试")
                continue

            # 输入照片名称
            while True:
                face_name = input("请输入照片名称（不能为空）：").strip()
                if face_name:
                    break
                print("❌ 名称不能为空，请重新输入")

            # 保存每个人脸（支持多个人脸，后缀加序号）
            save_success = False
            for i, det in enumerate(detections):
                x, y, w, h, score = det
                x, y, w, h = int(x), int(y), int(w), int(h)

                # 确保坐标在有效范围内
                h_img, w_img = frame.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w_img - x, w)
                h = min(h_img - y, h)

                if w > 0 and h > 0:
                    try:
                        # 扩大人脸框范围，包含更多头部区域
                        expand_factor = 1.5
                        new_w = int(w * expand_factor)
                        new_h = int(h * expand_factor)

                        # 计算新的中心点
                        center_x = x + w // 2
                        center_y = y + h // 2

                        # 计算新的坐标
                        new_x = max(0, center_x - new_w // 2)
                        new_y = max(0, center_y - new_h // 2)

                        # 确保不超出图像边界
                        new_w = min(w_img - new_x, new_w)
                        new_h = min(h_img - new_y, new_h)

                        # 使用dlib进行人脸对齐（输出112x112）
                        aligned_face = align_face(frame, (new_x, new_y, new_w, new_h))

                        # 生成文件名（多个人脸时加序号）
                        if len(detections) > 1:
                            aligned_filename = f"{face_name}_{i}.jpg"
                        else:
                            aligned_filename = f"{face_name}.jpg"

                        # 保存对齐后的112x112头像到face_database
                        aligned_path = os.path.join(opt.output, aligned_filename)
                        # 高质量保存（JPEG质量100）
                        cv2.imwrite(aligned_path, aligned_face, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        print(f"✅ 保存成功: {aligned_filename} (112x112) -> {aligned_path}")
                        save_success = True

                    except Exception as e:
                        print(f"❌ 人脸{i}对齐失败: {e}")

            if save_success:
                # 打印提示语并倒计时3秒退出
                print("\n📝 录取成功，3s后自动退出...")
                for i in range(3, 0, -1):
                    print(f"倒计时: {i}秒", end='\r')
                    time.sleep(1)
                break
            else:
                print("❌ 所有人脸保存失败，请重试")

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n===== 处理完成 ======")
    print(f"对齐人脸保存路径: {os.path.abspath(opt.output)}")
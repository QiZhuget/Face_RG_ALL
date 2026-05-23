import argparse
import cv2
import dlib
import numpy as np
import os
import time
import Function

def load_face_features(load_path='output/face_features.npz'):
    """加载离线生成的特征库"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"特征库文件不存在：{load_path}\n请先运行 face_feature_extractor.py 生成特征库")

    try:
        data = np.load(load_path, allow_pickle=True)
        names = data['names']
        features = data['features']
        face_database = {name: feat for name, feat in zip(names, features)}
        print(f"成功加载 {len(face_database)} 个人脸特征")
        return face_database
    except Exception as e:
        raise RuntimeError(f"加载特征库失败：{e}")


def match_face(face_feat, face_database, threshold=0.45):
    """匹配人脸到特征库"""
    if face_feat is None or not face_database:
        return False, "Unknown", 0.0

    face_feat = face_feat / (np.linalg.norm(face_feat) + 1e-12)
    best_similarity = 0.0
    best_name = "Unknown"

    for name, ref_feat in face_database.items():
        sim = Function.MobileFaceNetONNX.cosine(face_feat, ref_feat)
        if sim > best_similarity:
            best_similarity = sim
            best_name = name

    is_match = best_similarity > threshold
    if not is_match:
        best_name = "Unknown"

    return is_match, best_name, best_similarity


def clip_face_rect(rect, img_shape):
    """裁剪人脸框坐标，确保在图像范围内"""
    x, y, w, h = rect
    img_h, img_w = img_shape[:2]

    # 确保坐标有效
    x = max(0, int(x))
    y = max(0, int(y))
    w = min(img_w - x, int(w))
    h = min(img_h - y, int(h))

    return x, y, w, h


def align_face(img, face_rect, predictor):
    """
    使用dlib进行人脸对齐（门禁场景优化版）
    :param img: 原始图像 (BGR格式)
    :param face_rect: [x1, y1, x2, y2] YOLO检测框
    :param predictor: dlib形状预测器
    :return: 对齐后的人脸图像（112x112），失败返回None
    """
    # 转换为[x, y, w, h]格式
    x1, y1, x2, y2 = face_rect
    face_rect_xywh = [x1, y1, x2 - x1, y2 - y1]

    # 裁剪坐标，确保在图像范围内
    x, y, w, h = clip_face_rect(face_rect_xywh, img.shape)

    # 扩大人脸框范围，包含更多头部区域
    expand_factor = 1.5
    center_x = x + w // 2
    center_y = y + h // 2
    new_w = int(w * expand_factor)
    new_h = int(h * expand_factor)

    # 计算新的坐标
    new_x = max(0, center_x - new_w // 2)
    new_y = max(0, center_y - new_h // 2)

    # 确保不超出图像边界
    img_h, img_w = img.shape[:2]
    new_w = min(img_w - new_x, new_w)
    new_h = min(img_h - new_y, new_h)

    if new_w <= 0 or new_h <= 0:
        return None

    # 转换为dlib的矩形格式
    dlib_rect = dlib.rectangle(int(new_x), int(new_y), int(new_x + new_w), int(new_y + new_h))

    try:
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

        # 裁剪对齐后的头像
        aligned_face = aligned[int(new_y):int(new_y + new_h), int(new_x):int(new_x + new_w)]

        # 调整为112x112大小（MobileFaceNet标准输入）
        aligned_face = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LANCZOS4)

        # 检查是否需要翻转（解决镜像问题）
        if left_eye[0] > right_eye[0]:
            aligned_face = cv2.flip(aligned_face, 1)

        return aligned_face

    except Exception as e:
        # 对齐失败时返回None，降级使用原始裁剪
        return None


def realtime_match(feat_path,mobilefacenet_onnx,yolov8_onnx,dlib_predictor_path,threshold,yolov8_imgsz,yolov8_conf,yolov8_iou,device,camera_idx):
    # 1. 加载特征库
    face_database = load_face_features(feat_path)
    # 2. 初始化dlib人脸对齐模型
    try:
        dlib_predictor = dlib.shape_predictor(dlib_predictor_path)
        print(f"dlib模型加载成功：{dlib_predictor_path}")
    except Exception as e:
        raise RuntimeError(
            f"dlib模型加载失败")

    # 3. 初始化特征提取和检测模型
    print("初始化 MobileFaceNet 模型")
    providers = ['CPUExecutionProvider']
    if device != 'cpu' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    mfn = Function.MobileFaceNetONNX(onnx_path=mobilefacenet_onnx, providers=providers)

    yolo_model = Function.init_yolov8_face(yolo_onnx_path=yolov8_onnx, device=device)

    # 4. 打开摄像头
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头（索引：{camera_idx}）")

    print("\n开始实时人脸匹配（集成人脸对齐）")
    print("按 'q' 退出 ")
    fps_list = []

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面，退出")
            break
        frame_copy = frame.copy()

        # 5. 检测人脸
        face_detections = Function.detect_face_yolov8(yolo_model, frame, imgsz=yolov8_imgsz, conf=yolov8_conf,
                                                      iou=yolov8_iou)
        # 6. 匹配每个人脸
        for (x1, y1, x2, y2, score) in face_detections:

            # 第一步：绘制对齐扩大框（可视化）
            w = x2 - x1
            h = y2 - y1
            expand_factor = 1.5
            center_x = x1 + w // 2
            center_y = y1 + h // 2
            new_w = int(w * expand_factor)
            new_h = int(h * expand_factor)
            new_x = max(0, center_x - new_w // 2)
            new_y = max(0, center_y - new_h // 2)
            new_x2 = min(frame.shape[1], new_x + new_w)
            new_y2 = min(frame.shape[0], new_y + new_h)
            cv2.rectangle(frame, (int(new_x), int(new_y)), (int(new_x2), int(new_y2)), (255, 0, 0), 1, cv2.LINE_AA)

            # 第二步：执行人脸对齐
            aligned_face = align_face(frame_copy, [x1, y1, x2, y2], dlib_predictor)

            # 对齐失败时降级使用原始扩展裁剪
            if aligned_face is None:
                expand_w = 20
                expand_h = 30
                face_roi = frame_copy[
                           max(0, y1 - expand_h):min(frame.shape[0], y2 + expand_h),
                           max(0, x1 - expand_w):min(frame.shape[1], x2 + expand_w)
                           ]
            else:
                face_roi = aligned_face  # 对齐成功使用标准化人脸

            # ========== 特征提取 & 匹配 ==========
            face_feat = mfn.get_feature(face_roi)
            is_match, name, similarity = match_face(face_feat, face_database, threshold)

            # 7. 可视化标注
            color = (0, 255, 0) if is_match else (0, 0, 255)
            label = f"{name}: {similarity:.4f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Face: {score:.2f}", (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 8. 显示帧率和状态
        fps = 1 / (time.time() - start_time)
        fps_list.append(fps)
        avg_fps = np.mean(fps_list[-30:]) if fps_list else 0.0
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"DB: {len(face_database)} people", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 9. 显示画面
        cv2.imshow("Realtime Face Matcher (with Align)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n实时匹配结束")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='实时人脸匹配')
    # 特征库配置
    parser.add_argument('--feat_path', default='output/face_features.npz', help='离线特征库路径')
    # MobileFaceNet 配置
    parser.add_argument('--mobilefacenet_onnx', default='weights/model_mobilefacenet.onnx',
                        help='MobileFaceNet ONNX 路径')
    parser.add_argument('--threshold', type=float, default=0.73, help='相似度阈值')
    # YOLOv8-Face 配置
    parser.add_argument('--yolov8_onnx', default='weights/yolov8n-face-lindevs.onnx', help='YOLOv8-Face ONNX 路径')
    parser.add_argument('--yolov8_imgsz', type=int, default=640, help='Yolo 输入尺寸')
    parser.add_argument('--yolov8_conf', type=float, default=0.75, help='Yolo 置信度阈值')
    parser.add_argument('--yolov8_iou', type=float, default=0.7, help='Yolo NMS IoU 阈值')
    # dlib对齐配置
    parser.add_argument('--dlib_predictor', default='weights/shape_predictor_68_face_landmarks.dat',
                        help='dlib形状预测器路径')
    # 其他配置
    parser.add_argument('--device', default='cpu', help='设备 (cpu/0)')
    parser.add_argument('--camera', type=int, default='0', help='摄像头索引')
   # parser.add_argument('--camera', type=str, default='http://192.168.137.14:4747/video', help='摄像头索引/手机IP地址')
    opt = parser.parse_args()

    try:
        realtime_match(
            feat_path=opt.feat_path,
            mobilefacenet_onnx=opt.mobilefacenet_onnx,
            yolov8_onnx=opt.yolov8_onnx,
            dlib_predictor_path=opt.dlib_predictor,
            threshold=opt.threshold,
            yolov8_imgsz=opt.yolov8_imgsz,
            yolov8_conf=opt.yolov8_conf,
            yolov8_iou=opt.yolov8_iou,
            device=opt.device,
            camera_idx=opt.camera
        )
    except Exception as e:
        print(f"程序出错：{e}")
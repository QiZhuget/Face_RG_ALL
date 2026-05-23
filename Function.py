import cv2
import numpy as np
import onnxruntime as ort
import dlib

def align_face(img, face_rect, predictor=None):
    """使用dlib进行人脸对齐

    支持两种调用方式：
    1. align_face(face_roi, predictor) - 仅传入人脸图像和预测器
    2. align_face(img, face_rect, predictor) - 传入完整图像、边界框和预测器
    """
    try:
        # 判断调用方式
        if predictor is None and isinstance(face_rect, dlib.shape_predictor):
            # 方式1：align_face(face_roi, predictor)
            predictor = face_rect
            face_roi = img
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            shape = predictor(gray, dlib.rectangle(0, 0, face_roi.shape[1], face_roi.shape[0]))
        else:
            # 方式2：align_face(img, face_rect, predictor)
            face_roi = img
            x1, y1, x2, y2 = face_rect
            w = x2 - x1
            h = y2 - y1

            expand_factor = 1.5
            center_x = x1 + w // 2
            center_y = y1 + h // 2
            new_w = int(w * expand_factor)
            new_h = int(h * expand_factor)
            new_x = max(0, center_x - new_w // 2)
            new_y = max(0, center_y - new_h // 2)
            new_x2 = min(face_roi.shape[1], new_x + new_w)
            new_y2 = min(face_roi.shape[0], new_y + new_h)

            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            dlib_rect = dlib.rectangle(int(new_x), int(new_y), int(new_x2), int(new_y2))
            shape = predictor(gray, dlib_rect)

        # 提取关键点
        landmarks = []
        for i in range(68):
            landmarks.append((shape.part(i).x, shape.part(i).y))

        # 计算仿射变换
        src_pts = np.array([
            landmarks[30],  # 鼻尖
            landmarks[8],   # 下巴
            landmarks[36],  # 左眼左角
            landmarks[45],  # 右眼右角
            landmarks[48],  # 左嘴角
            landmarks[54]   # 右嘴角
        ], dtype=np.float32)

        dst_pts = np.array([
            [56, 48],     # 鼻尖
            [56, 104],    # 下巴
            [30, 48],     # 左眼左角
            [82, 48],     # 右眼右角
            [38, 80],     # 左嘴角
            [74, 80]      # 右嘴角
        ], dtype=np.float32)

        # 计算变换矩阵
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            return None

        # 应用变换
        aligned_face = cv2.warpAffine(face_roi, M, (112, 112))
        return aligned_face
    except Exception as e:
        print(f"人脸对齐失败：{e}")
        return None

class MobileFaceNetONNX:
    """MobileFaceNet 特征提取类"""

    def __init__(self, onnx_path='weights/model_mobilefacenet.onnx', providers=None):
        self.providers = providers or ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (112, 112))
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img.transpose(2, 0, 1)[None, ...]

    def get_feature(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return None
        try:
            inp = self._preprocess(img_bgr)
            feat = self.session.run(None, {self.input_name: inp})[0]
            return feat.flatten()
        except Exception as e:
            print(f"特征提取失败：{e}")
            return None

    @staticmethod
    def cosine(feat1, feat2):
        """计算余弦相似度"""
        if feat1 is None or feat2 is None:
            return 0.0
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-12)


def init_yolov8_face(yolo_onnx_path='weights/yolov8n-face-lindevs.onnx', device='cpu'):
    """初始化YOLOv8-Face检测器"""
    model = cv2.dnn.readNetFromONNX(yolo_onnx_path)
    if device != 'cpu' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return model

def detect_face_yolov8(model, frame, imgsz=640, conf=0.75, iou=0.7):
    """YOLOv8-Face检测人脸"""
    height, width, _ = frame.shape
    length = max((height, width))
    scale = length / imgsz

    blob = np.zeros((length, length, 3), np.uint8)
    blob[0:height, 0:width] = frame
    blob = cv2.dnn.blobFromImage(blob, scalefactor=1 / 255, size=(imgsz, imgsz), swapRB=True)

    model.setInput(blob)
    outputs = model.forward()
    outputs = cv2.transpose(outputs[0])

    boxes = []
    scores = []
    for i in range(outputs.shape[0]):
        max_score = float(np.amax(outputs[i][4:]))
        if max_score >= conf:
            x = (outputs[i][0] - (0.5 * outputs[i][2])) * scale
            y = (outputs[i][1] - (0.5 * outputs[i][3])) * scale
            w = outputs[i][2] * scale
            h = outputs[i][3] * scale
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(width, int(round(x + w)))
            y2 = min(height, int(round(y + h)))
            boxes.append([x1, y1, x2, y2])
            scores.append(max_score)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    results = []
    if len(indices) > 0:
        indices = indices.flatten() if isinstance(indices, (np.ndarray, list)) else [indices]
        for idx in indices:
            x1, y1, x2, y2 = boxes[idx]
            results.append([x1, y1, x2, y2, scores[idx]])
    return results
import argparse
import cv2
import dlib
import numpy as np
import os
import time
import Function
import tkinter as tk
from tkinter import ttk, Frame, Button, Text, Scrollbar, messagebox, simpledialog, Entry, Listbox, Label
from PIL import ImageTk, Image
import sys
import threading
from queue import Queue
import glob
from pathlib import Path

# -------------------------- 日志重定向 --------------------------
class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


# ====================== 照片转特征（全量生成） ======================
def save_face_features(face_database, save_path='output/face_features.npz'):
    if not face_database:
        print("无有效特征可保存")
        return
    names = list(face_database.keys())
    features = np.array(list(face_database.values()))
    np.savez_compressed(save_path, names=names, features=features)
    print(f"特征库已保存到：{save_path}")
    print(f"   共保存 {len(names)} 个人脸特征：{', '.join(names)}")


def extract_features_from_folder(
        face_db_dir='face_database',
        onnx_path='weights/model_mobilefacenet.onnx',
        save_path='output/face_features.npz',
        device='cpu'
):
    providers = ['CPUExecutionProvider']
    if device != 'cpu' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    mfn = Function.MobileFaceNetONNX(onnx_path=onnx_path, providers=providers)

    if not os.path.exists(face_db_dir):
        os.makedirs(face_db_dir)
        return

    face_database = {}
    valid_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    img_paths = glob.glob(os.path.join(face_db_dir, '*'))

    for img_path in img_paths:
        if Path(img_path).suffix.lower() not in valid_ext:
            continue
        name = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None:
            continue
        feat = mfn.get_feature(img)
        if feat is None:
            continue
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        face_database[name] = feat

    save_face_features(face_database, save_path)


# -------------------------- 原功能代码 --------------------------
def load_face_features(load_path='output/face_features.npz'):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"特征库文件不存在：{load_path}")
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
    return is_match, best_name, best_similarity

def clip_face_rect(rect, img_shape):
    x, y, w, h = rect
    img_h, img_w = img_shape[:2]
    x = max(0, int(x))
    y = max(0, int(y))
    w = min(img_w - x, int(w))
    h = min(img_h - y, int(h))
    return x, y, w, h

def align_face(img, face_rect, predictor):
    x1, y1, x2, y2 = face_rect
    face_rect_xywh = [x1, y1, x2 - x1, y2 - y1]
    x, y, w, h = clip_face_rect(face_rect_xywh, img.shape)
    expand_factor = 1.5
    center_x = x + w // 2
    center_y = y + h // 2
    new_w = int(w * expand_factor)
    new_h = int(h * expand_factor)
    new_x = max(0, center_x - new_w // 2)
    new_y = max(0, center_y - new_h // 2)
    img_h, img_w = img.shape[:2]
    new_w = min(img_w - new_x, new_w)
    new_h = min(img_h - new_y, new_h)
    if new_w <= 0 or new_h <= 0:
        return None
    dlib_rect = dlib.rectangle(int(new_x), int(new_y), int(new_x + new_w), int(new_y + new_h))
    try:
        shape = predictor(img, dlib_rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
        RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
        left_eye = landmarks[LEFT_EYE_INDICES].mean(axis=0)
        right_eye = landmarks[RIGHT_EYE_INDICES].mean(axis=0)
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        aligned_face = aligned[int(new_y):int(new_y + new_h), int(new_x):int(new_x + new_w)]
        aligned_face = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LANCZOS4)
        if left_eye[0] > right_eye[0]:
            aligned_face = cv2.flip(aligned_face, 1)
        return aligned_face
    except Exception as e:
        return None


# ====================== 拍照保存 ======================
def capture_preview_save_face(frame, yolo_model, dlib_predictor, app, save_dir="face_database"):
    os.makedirs(save_dir, exist_ok=True)

    def detect_face(img, model, imgsz=640, conf_threshold=0.75, iou_threshold=0.7):
        height, width, _ = img.shape
        length = max((height, width))
        scale = length / imgsz
        blob = np.zeros((length, length, 3), np.uint8)
        blob[0:height, 0:width] = img
        blob = cv2.dnn.blobFromImage(blob, scalefactor=1 / 255, size=(imgsz, imgsz), swapRB=True)
        model.setInput(blob)
        outputs = model.forward()
        outputs = cv2.transpose(outputs[0])
        boxes = []
        scores = []
        for i in range(outputs.shape[0]):
            max_score = float(np.amax(outputs[i][4:]))
            if max_score >= conf_threshold:
                boxes.append([(outputs[i][0] - 0.5 * outputs[i][2]) * scale,
                              (outputs[i][1] - 0.5 * outputs[i][3]) * scale,
                              outputs[i][2] * scale, outputs[i][3] * scale])
                scores.append(max_score)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        detections = []
        if indices is not None and len(indices) > 0:
            for idx in indices:
                idx = idx if isinstance(idx, (int, np.integer)) else idx[0]
                x, y, w, h = boxes[idx]
                detections.append([x, y, w, h, scores[idx]])
        return detections

    # ====================== 这里统一使用和实时识别一样的扩大框 + 对齐逻辑 ======================
    def align_face_save(img, face_rect):
        x1, y1, x2, y2 = face_rect
        # 转为 xywh 格式
        face_rect_xywh = [x1, y1, x2 - x1, y2 - y1]

        # 裁剪越界处理
        def clip_face_rect(rect, img_shape):
            x, y, w, h = rect
            img_h, img_w = img_shape[:2]
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(img_w - x, int(w))
            h = min(img_h - y, int(h))
            return x, y, w, h

        x, y, w, h = clip_face_rect(face_rect_xywh, img.shape)

        # ====================== 扩大人脸框 1.5 倍 ======================
        expand_factor = 1.5
        center_x = x + w // 2
        center_y = y + h // 2
        new_w = int(w * expand_factor)
        new_h = int(h * expand_factor)
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        img_h, img_w = img.shape[:2]
        new_w = min(img_w - new_x, new_w)
        new_h = min(img_h - new_y, new_h)

        if new_w <= 0 or new_h <= 0:
            return None

        # 使用扩大后的框做关键点检测
        dlib_rect = dlib.rectangle(int(new_x), int(new_y), int(new_x + new_w), int(new_y + new_h))
        shape = dlib_predictor(img, dlib_rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
        RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
        left_eye = landmarks[LEFT_EYE_INDICES].mean(axis=0)
        right_eye = landmarks[RIGHT_EYE_INDICES].mean(axis=0)

        # 旋转对齐
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # 截取扩大后的人脸区域
        aligned_face = aligned[int(new_y):int(new_y + new_h), int(new_x):int(new_x + new_w)]
        aligned_face = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LANCZOS4)

        # 左右眼位置纠正
        if left_eye[0] > right_eye[0]:
            aligned_face = cv2.flip(aligned_face, 1)
        return aligned_face

    detections = detect_face(frame, yolo_model)
    if len(detections) == 0:
        messagebox.showwarning("提示", "未检测到人脸！")
        return

    best_det = max(detections, key=lambda d: (d[2] * d[3]))
    x, y, w, h, score = best_det
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)  # 转为对角坐标

    try:
        aligned_face = align_face_save(frame, (x1, y1, x2, y2))

        preview_win = tk.Toplevel()
        preview_win.title("人脸捕获确认")
        preview_win.geometry("320x400")
        preview_win.resizable(False, False)
        preview_win.attributes('-topmost', True)

        face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tk = ImageTk.PhotoImage(face_pil)

        img_label = Label(preview_win, image=face_tk)
        img_label.image = face_tk
        img_label.pack(pady=30)

        Label(preview_win, text="请输入姓名：", font=("微软雅黑", 11)).pack(pady=5)
        name_entry = Entry(preview_win, font=("微软雅黑", 12), width=18, justify="center")
        name_entry.pack(pady=5)
        name_entry.focus()

        def do_save():
            face_name = name_entry.get().strip()
            if not face_name:
                messagebox.showwarning("提示", "姓名不能为空！", parent=preview_win)
                return
            save_path = os.path.join(save_dir, f"{face_name}.jpg")
            cv2.imwrite(save_path, aligned_face, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"✅ 保存成功：{save_path}")

            extract_features_from_folder(
                face_db_dir=save_dir,
                onnx_path=app.opt.mobilefacenet_onnx,
                save_path=app.opt.feat_path,
                device=app.opt.device
            )
            app.face_database = load_face_features(app.opt.feat_path)
            messagebox.showinfo("成功", f"【{face_name}】已录入", parent=preview_win)
            preview_win.destroy()

        confirm_btn = Button(preview_win, text="确认保存", font=("微软雅黑", 11), width=12, command=do_save)
        confirm_btn.pack(pady=15)
        name_entry.bind('<Return>', lambda e: do_save())
        preview_win.wait_window()

    except Exception as e:
        print(f"❌ 处理失败：{e}")
        messagebox.showerror("错误", "人脸捕获保存失败！")


# ====================== 删除人脸窗口 ======================
def delete_face_window(app):
    save_dir = "face_database"
    valid_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    files = [f for f in os.listdir(save_dir) if Path(f).suffix.lower() in valid_ext]

    win = tk.Toplevel()
    win.title("删除人脸")
    win.geometry("450x500")
    win.resizable(False, False)
    win.attributes('-topmost', True)

    Label(win, text="选择要删除的人脸", font=("微软雅黑", 13)).pack(pady=10)

    listbox = Listbox(win, font=("微软雅黑", 12), width=35, height=12)
    listbox.pack(pady=5, padx=20)

    for i, name in enumerate(files):
        display_name = Path(name).stem
        listbox.insert(tk.END, f" {i+1:2d} → {display_name}")

    def do_delete():
        selected = listbox.curselection()
        if not selected:
            messagebox.showwarning("提示", "请选择一个要删除的人脸", parent=win)
            return
        idx = selected[0]
        del_file = os.path.join(save_dir, files[idx])
        name_stem = Path(files[idx]).stem

        if messagebox.askyesno("确认", f"确定要删除【{name_stem}】吗？", parent=win):
            try:
                os.remove(del_file)
                extract_features_from_folder(
                    face_db_dir=save_dir,
                    onnx_path=app.opt.mobilefacenet_onnx,
                    save_path=app.opt.feat_path,
                    device=app.opt.device
                )
                app.face_database = load_face_features(app.opt.feat_path)
                messagebox.showinfo("成功", f"【{name_stem}】已删除\n特征库已更新", parent=win)
                win.destroy()
            except:
                messagebox.showerror("错误", "删除失败", parent=win)

    Button(win, text="删除选中项", font=("微软雅黑", 11), bg="#ff4444", fg="white", width=15, command=do_delete).pack(pady=20)


# -------------------------- GUI 主界面 --------------------------
class FaceRecognitionGUI:
    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        self.root.title("实时人脸识别系统")
        self.root.geometry("900x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.running = True
        self.in_settings = False
        self.last_time = time.time()
        self.fps_list = []
        self.avg_fps = 0.0
        self.frame_queue = Queue(maxsize=2)
        self.current_frame = None

        self.face_database = load_face_features(opt.feat_path)
        self.dlib_predictor = dlib.shape_predictor(opt.dlib_predictor)
        providers = ['CPUExecutionProvider']
        if opt.device != 'cpu' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.mfn = Function.MobileFaceNetONNX(onnx_path=opt.mobilefacenet_onnx, providers=providers)
        self.yolo_model = Function.init_yolov8_face(yolo_onnx_path=opt.yolov8_onnx, device=opt.device)
        self.yolo_model_raw = cv2.dnn.readNetFromONNX(opt.yolov8_onnx)

        self.cap = cv2.VideoCapture(opt.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.fps_label = Label(root, text="FPS: --", font=("Arial", 12, "bold"), fg="blue")
        self.fps_label.place(x=20, y=10)

        self.video_frame = Frame(root, width=640, height=480)
        self.video_frame.place(x=130, y=40)
        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        self.btn_frame = Frame(root)
        self.btn_frame.place(x=350, y=540)
        self.setting_btn = Button(self.btn_frame, text="设置", width=10, command=self.show_settings)
        self.setting_btn.grid(row=0, column=0, padx=10)
        self.exit_btn = Button(self.btn_frame, text="退出", width=10, command=self.on_exit)
        self.exit_btn.grid(row=0, column=1, padx=10)

        self.back_btn = Button(self.btn_frame, text="返回主界面", width=15, command=self.hide_settings)
        self.capture_btn = Button(self.btn_frame, text="捕获人脸照片", width=15, command=self.capture_face)
        self.delete_btn = Button(self.btn_frame, text="删除人脸", width=15, command=lambda: delete_face_window(self))

        self.log_frame = Frame(root)
        self.log_frame.place(x=100, y=580, width=700, height=80)
        self.log_text = Text(self.log_frame)
        self.log_scroll = Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        sys.stdout = TextRedirector(self.log_text)
        print("系统初始化完成")

        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.update_frame()
        self.update_fps()

    def capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头")
                break
            self.current_frame = frame.copy()

            if self.in_settings:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                time.sleep(0.01)
                continue

            frame_copy = frame.copy()
            face_detections = Function.detect_face_yolov8(
                self.yolo_model, frame,
                imgsz=self.opt.yolov8_imgsz,
                conf=self.opt.yolov8_conf,
                iou=self.opt.yolov8_iou
            )

            for (x1, y1, x2, y2, score) in face_detections:
                aligned_face = align_face(frame_copy, [x1, y1, x2, y2], self.dlib_predictor)
                if aligned_face is not None:
                    feat = self.mfn.get_feature(aligned_face)
                    match, name, sim = match_face(feat, self.face_database, self.opt.threshold)

                    if match:
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(frame, f"{name}:{sim:.2f}", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time + 1e-6)
            self.last_time = current_time
            self.fps_list.append(fps)
            if len(self.fps_list) > 20:
                self.fps_list.pop(0)
            self.avg_fps = np.mean(self.fps_list)

            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())

    def update_frame(self):
        if not self.running:
            return
        frame = None
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            self.imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.configure(image=self.imgtk)
        self.root.after(25, self.update_frame)

    def update_fps(self):
        self.fps_label.config(text=f"FPS: {self.avg_fps:.1f}")
        self.root.after(200, self.update_fps)

    def capture_face(self):
        if self.current_frame is None:
            messagebox.showwarning("提示", "未获取到摄像头画面")
            return
        threading.Thread(target=lambda: capture_preview_save_face(
            self.current_frame, self.yolo_model_raw, self.dlib_predictor, self
        ), daemon=True).start()

    def show_settings(self):
        self.in_settings = True
        self.setting_btn.grid_forget()
        self.back_btn.grid(row=0, column=0, padx=5)
        self.capture_btn.grid(row=0, column=1, padx=5)
        self.delete_btn.grid(row=0, column=2, padx=5)

    def hide_settings(self):
        self.in_settings = False
        self.back_btn.grid_forget()
        self.capture_btn.grid_forget()
        self.delete_btn.grid_forget()
        self.setting_btn.grid(row=0, column=0, padx=10)

    def on_exit(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()


# -------------------------- 主程序 --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='实时人脸匹配(GUI)')
    parser.add_argument('--feat_path', default='output/face_features.npz')
    parser.add_argument('--mobilefacenet_onnx', default='weights/model_mobilefacenet.onnx')
    parser.add_argument('--threshold', type=float, default=0.73)
    parser.add_argument('--yolov8_onnx', default='weights/yolov8n-face-lindevs.onnx')
    parser.add_argument('--yolov8_imgsz', type=int, default=640)
    parser.add_argument('--yolov8_conf', type=float, default=0.75)
    parser.add_argument('--yolov8_iou', type=float, default=0.7)
    parser.add_argument('--dlib_predictor', default='weights/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--camera', type=int, default=0)
    opt = parser.parse_args()

    root = tk.Tk()
    app = FaceRecognitionGUI(root, opt)
    root.mainloop()

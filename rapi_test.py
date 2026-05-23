#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
import os
import dlib
import onnxruntime
import time
import tkinter as tk
from tkinter import Frame, Button, Text, Scrollbar, messagebox, Entry, Listbox, Label
from PIL import ImageTk, Image
import sys
import threading
from queue import Queue
import glob
from pathlib import Path

# -------------------------- Log redirection --------------------------
class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
    def flush(self):
        pass

# ====================== Original alignment function ======================
def align_face(img, landmarks, desired_width=112):
    """Face alignment using eye positions"""
    right_eye = np.array([landmarks.part(0).x, landmarks.part(0).y], dtype=np.float32)
    left_eye = np.array([landmarks.part(2).x, landmarks.part(2).y], dtype=np.float32)

    desired_right = np.array([desired_width * 0.7, desired_width * 0.46], dtype=np.float32)
    desired_left = np.array([desired_width * 0.3, desired_width * 0.46], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(np.array([right_eye, left_eye]),
                                       np.array([desired_right, desired_left]))
    aligned = cv2.warpAffine(img, M, (desired_width, desired_width), flags=cv2.INTER_CUBIC)
    return aligned

# ====================== YOLOv8 face detection =======================
def preprocess_image(img, imgsz=320):
    h, w = img.shape[:2]
    scale = imgsz / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    blob = canvas.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, (pad_x, pad_y)

def postprocess(outputs, conf_thresh, iou_thresh, img_shape, scale_factor, pad):
    if outputs.ndim == 3:
        outputs = outputs[0]
    if outputs.shape[0] < outputs.shape[1]:
        outputs = outputs.T

    boxes_xywh = []
    scores = []
    for pred in outputs:
        max_score = np.amax(pred[4:])
        if max_score < conf_thresh:
            continue
        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
        boxes_xywh.append([cx, cy, w, h])
        scores.append(max_score)

    if not boxes_xywh:
        return []

    boxes_xywh_fill = np.array(boxes_xywh)
    boxes_xywh_fill[:, 0] -= boxes_xywh_fill[:, 2] / 2
    boxes_xywh_fill[:, 1] -= boxes_xywh_fill[:, 3] / 2

    indices = cv2.dnn.NMSBoxes(boxes_xywh_fill.tolist(), scores, conf_thresh, iou_thresh)
    if indices is None or len(indices) == 0:
        return []
    if isinstance(indices[0], (list, np.ndarray)):
        indices = [idx[0] for idx in indices]
    else:
        indices = indices.flatten().tolist()

    h, w = img_shape[:2]
    detections = []
    for idx in indices:
        x_fill, y_fill, w_fill, h_fill = boxes_xywh_fill[idx]
        x_orig = (x_fill - pad[0]) / scale_factor
        y_orig = (y_fill - pad[1]) / scale_factor
        w_orig = w_fill / scale_factor
        h_orig = h_fill / scale_factor
        x1 = max(0, int(x_orig))
        y1 = max(0, int(y_orig))
        x2 = min(w, int(x_orig + w_orig))
        y2 = min(h, int(y_orig + h_orig))
        if x2 > x1 and y2 > y1:
            detections.append((x1, y1, x2, y2, scores[idx]))
    return detections

# ====================== MobileFaceNet feature extraction ======================
class MobileFaceNetONNX:
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def get_feature(self, face_img):
        if face_img is None:
            return None
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        feat = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        return feat

    @staticmethod
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

# ====================== Face database management =======================
def save_face_features(face_database, save_path='output/face_features.npz'):
    if not face_database:
        print("No features to save")
        return
    names = list(face_database.keys())
    features = np.array(list(face_database.values()))
    np.savez_compressed(save_path, names=names, features=features)
    print(f"Features saved to: {save_path}")
    print(f"   Saved {len(names)} faces: {', '.join(names)}")

def load_face_features(load_path='output/face_features.npz'):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Feature file not found: {load_path}")
    data = np.load(load_path, allow_pickle=True)
    names = data['names']
    features = data['features']
    face_database = {name: feat for name, feat in zip(names, features)}
    print(f"Loaded {len(face_database)} face features")
    return face_database

def extract_features_from_folder(face_db_dir='face_database', onnx_path='weights/model_mobilefacenet.onnx',
                                 save_path='output/face_features.npz'):
    mfn = MobileFaceNetONNX(onnx_path=onnx_path)

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

def match_face(face_feat, face_database, threshold=0.45):
    if face_feat is None or not face_database:
        return False, "Unknown", 0.0
    face_feat = face_feat / (np.linalg.norm(face_feat) + 1e-12)
    best_sim = 0.0
    best_name = "Unknown"
    for name, ref_feat in face_database.items():
        sim = MobileFaceNetONNX.cosine(face_feat, ref_feat)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    is_match = best_sim > threshold
    return is_match, best_name, best_sim

# ====================== Capture and save face ======================
def capture_and_save_face(frame, yolo_session, dlib_predictor, app, save_dir="face_database",
                          imgsz=640, conf=0.75, iou=0.7):
    os.makedirs(save_dir, exist_ok=True)
    blob, scale, pad = preprocess_image(frame, imgsz)
    outputs = yolo_session.run(None, {yolo_session.get_inputs()[0].name: blob})[0]
    detections = postprocess(outputs, conf, iou, frame.shape, scale, pad)
    if len(detections) == 0:
        messagebox.showwarning("Warning", "No face detected")
        return

    best_det = max(detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
    x1, y1, x2, y2, _ = best_det
    dlib_rect = dlib.rectangle(x1, y1, x2, y2)
    try:
        landmarks = dlib_predictor(frame, dlib_rect)
        aligned_face = align_face(frame, landmarks)
    except Exception as e:
        messagebox.showerror("Error", f"Face alignment failed: {e}")
        return

    preview_win = tk.Toplevel()
    preview_win.title("Confirm Face Capture")
    preview_win.geometry("320x400")
    preview_win.resizable(False, False)
    preview_win.attributes('-topmost', True)

    # Handle window close button (X)
    def on_preview_close():
        preview_win.destroy()
    preview_win.protocol("WM_DELETE_WINDOW", on_preview_close)

    # Center window
    preview_win.update_idletasks()
    width = preview_win.winfo_width()
    height = preview_win.winfo_height()
    screen_width = preview_win.winfo_screenwidth()
    screen_height = preview_win.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    preview_win.geometry(f"+{x}+{y}")

    face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_tk = ImageTk.PhotoImage(face_pil)

    img_label = Label(preview_win, image=face_tk)
    img_label.image = face_tk
    img_label.pack(pady=30)

    Label(preview_win, text="Enter name:", font=("Microsoft YaHei", 11)).pack(pady=5)
    name_entry = Entry(preview_win, font=("Microsoft YaHei", 12), width=18, justify="center")
    name_entry.pack(pady=5)
    name_entry.focus()

    def do_save():
        face_name = name_entry.get().strip()
        if not face_name:
            messagebox.showwarning("Warning", "Name cannot be empty", parent=preview_win)
            return
        save_path = os.path.join(save_dir, f"{face_name}.jpg")
        cv2.imwrite(save_path, aligned_face, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"[OK] Saved: {save_path}")

        extract_features_from_folder(
            face_db_dir=save_dir,
            onnx_path=app.opt.mobilefacenet_onnx,
            save_path=app.opt.feat_path
        )
        app.face_database = load_face_features(app.opt.feat_path)
        messagebox.showinfo("Success", f"{face_name} added", parent=preview_win)
        preview_win.destroy()

    Button(preview_win, text="Save", font=("Microsoft YaHei", 11), width=12, command=do_save).pack(pady=15)
    name_entry.bind('<Return>', lambda e: do_save())
    preview_win.wait_window()

# ====================== Delete face window ======================
def delete_face_window(app):
    save_dir = "face_database"
    valid_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    files = [f for f in os.listdir(save_dir) if Path(f).suffix.lower() in valid_ext]
    if not files:
        messagebox.showinfo("Info", "Face database empty")
        return

    win = tk.Toplevel()
    win.title("Delete Face")
    win.geometry("450x500")
    win.resizable(False, False)
    win.attributes('-topmost', True)

    Label(win, text="Select face to delete", font=("Microsoft YaHei", 13)).pack(pady=10)
    listbox = Listbox(win, font=("Microsoft YaHei", 12), width=35, height=12)
    listbox.pack(pady=5, padx=20)

    for i, name in enumerate(files):
        display_name = Path(name).stem
        listbox.insert(tk.END, f" {i+1:2d}   -> {display_name}")

    def do_delete():
        selected = listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a face", parent=win)
            return
        idx = selected[0]
        del_file = os.path.join(save_dir, files[idx])
        name_stem = Path(files[idx]).stem

        if messagebox.askyesno("Confirm", f"Delete {name_stem}?", parent=win):
            try:
                os.remove(del_file)
                extract_features_from_folder(
                    face_db_dir=save_dir,
                    onnx_path=app.opt.mobilefacenet_onnx,
                    save_path=app.opt.feat_path
                )
                app.face_database = load_face_features(app.opt.feat_path)
                messagebox.showinfo("Success", f"{name_stem} deleted\nDatabase updated", parent=win)
                win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Deletion failed: {e}", parent=win)

    Button(win, text="Delete Selected", font=("Microsoft YaHei", 11), bg="#ff4444", fg="white", width=15, command=do_delete).pack(pady=20)

# ====================== Main GUI (no servo) ======================
class FaceRecognitionGUI:
    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        self.root.title("Real-time Face Recognition System")
        self.root.geometry("1200x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.running = True
        self.in_settings = False
        self.last_time = time.time()
        self.fps_list = []
        self.avg_fps = 0.0
        self.frame_queue = Queue(maxsize=2)
        self.current_frame = None

        # Load models and database
        self.face_database = load_face_features(opt.feat_path)
        self.dlib_predictor = dlib.shape_predictor(opt.dlib_predictor)
        self.mfn = MobileFaceNetONNX(onnx_path=opt.mobilefacenet_onnx)
        self.yolo_session = onnxruntime.InferenceSession(opt.yolov8_onnx, providers=['CPUExecutionProvider'])

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ---------------------- GUI layout ----------------------
        self.left_frame = Frame(root, width=200, bg='#f0f0f0')
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_frame.pack_propagate(False)

        self.center_frame = Frame(self.left_frame, bg='#f0f0f0')
        self.center_frame.pack(expand=True)

        self.setting_btn = Button(self.center_frame, text="Settings", width=12, command=self.show_settings)
        self.exit_btn = Button(self.center_frame, text="Exit", width=12, command=self.on_exit)
        self.back_btn = Button(self.center_frame, text="Back", width=12, command=self.hide_settings)
        self.capture_btn = Button(self.center_frame, text="Capture Face", width=12, command=self.capture_face)
        self.delete_btn = Button(self.center_frame, text="Delete Face", width=12, command=lambda: delete_face_window(self))

        self.setting_btn.pack(pady=8)
        self.exit_btn.pack(pady=8)
        self.back_btn.pack_forget()
        self.capture_btn.pack_forget()
        self.delete_btn.pack_forget()

        self.right_frame = Frame(root, bg='white')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_frame = Frame(self.right_frame, width=640, height=480)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=20)
        self.video_frame.pack_propagate(False)
        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        self.fps_label = Label(self.video_frame, text="FPS: --", font=("Arial", 12, "bold"), fg="blue", bg="white")
        self.fps_label.place(x=10, y=10)

        self.log_frame = Frame(self.right_frame, width=350, height=480, bg='#f8f8f8')
        self.log_frame.pack(side=tk.RIGHT, padx=10, pady=20, fill=tk.BOTH, expand=True)
        self.log_frame.pack_propagate(False)

        self.log_text = Text(self.log_frame, wrap=tk.WORD, bg='#ffffff')
        self.log_scroll = Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        sys.stdout = TextRedirector(self.log_text)
        print("System initialized")

        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.update_frame()
        self.update_fps()

    # ---------- Camera capture and recognition thread ----------
    def capture_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera read failed, stopping capture loop")
                    break
                self.current_frame = frame.copy()

                if self.in_settings:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                    time.sleep(0.01)
                    continue

                blob, scale, pad = preprocess_image(frame, self.opt.yolov8_imgsz)
                outputs = self.yolo_session.run(None, {self.yolo_session.get_inputs()[0].name: blob})[0]
                detections = postprocess(outputs, self.opt.yolov8_conf, self.opt.yolov8_iou,
                                         frame.shape, scale, pad)

                for (x1, y1, x2, y2, score) in detections:
                    dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                    try:
                        landmarks = self.dlib_predictor(frame, dlib_rect)
                        aligned_face = align_face(frame, landmarks)
                        feat = self.mfn.get_feature(aligned_face)
                        match, name, sim = match_face(feat, self.face_database, self.opt.threshold)

                        if match:
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                            cv2.putText(frame, f"{name}:{sim:.2f}", (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            color = (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    except Exception as e:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 1e-6)
                self.last_time = current_time
                self.fps_list.append(fps)
                if len(self.fps_list) > 20:
                    self.fps_list.pop(0)
                self.avg_fps = np.mean(self.fps_list)

                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

            except Exception as e:
                print(f"Capture loop exception: {e}")
                break

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

    # ---------- Capture face (no extra thread) ----------
    def capture_face(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No camera frame")
            return
        capture_and_save_face(
            self.current_frame, self.yolo_session, self.dlib_predictor, self,
            save_dir="face_database",
            imgsz=self.opt.yolov8_imgsz,
            conf=self.opt.yolov8_conf,
            iou=self.opt.yolov8_iou
        )

    def show_settings(self):
        self.in_settings = True
        self.setting_btn.pack_forget()
        self.exit_btn.pack_forget()
        self.back_btn.pack(pady=8)
        self.capture_btn.pack(pady=8)
        self.delete_btn.pack(pady=8)

    def hide_settings(self):
        self.in_settings = False
        self.back_btn.pack_forget()
        self.capture_btn.pack_forget()
        self.delete_btn.pack_forget()
        self.setting_btn.pack(pady=8)
        self.exit_btn.pack(pady=8)

    # ---------- Clean exit ----------
    def on_exit(self):
        self.running = False

        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            if self.capture_thread.is_alive():
                print("Warning: capture thread did not terminate normally")

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break

        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()

# -------------------------- Main --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Face Recognition System')
    parser.add_argument('--feat_path', default='output/face_features.npz')
    parser.add_argument('--mobilefacenet_onnx', default='weights/model_mobilefacenet_int8.onnx')
    parser.add_argument('--threshold', type=float, default=0.73)
    parser.add_argument('--yolov8_onnx', default='weights/yolov8n-face_320-int8.onnx')
    parser.add_argument('--yolov8_imgsz', type=int, default=320)
    parser.add_argument('--yolov8_conf', type=float, default=0.75)
    parser.add_argument('--yolov8_iou', type=float, default=0.7)
    parser.add_argument('--dlib_predictor', default='weights/shape_predictor_5_face_landmarks.dat')
    parser.add_argument('--camera', type=int, default=0)
    opt = parser.parse_args()

    root = tk.Tk()
    app = FaceRecognitionGUI(root, opt)
    root.mainloop()
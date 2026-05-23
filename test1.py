import argparse
import cv2
import numpy as np
import onnxruntime
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/yolov8n-face_320-int8.onnx')
    parser.add_argument('--source', default=0)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--conf', type=float, default=0.75)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--device', default='cpu')
    opt = parser.parse_args()

    providers = ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(opt.weights, providers=providers)
    input_name = session.get_inputs()[0].name

    # ===================== PC 专用：去掉树莓派驱动 =====================
    cap = cv2.VideoCapture(opt.source)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    # PC 性能强，可设 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev_time = time.perf_counter()
    fps = 0.0

    print("Camera started, press q to quit")

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        current_time = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (current_time - prev_time)
        prev_time = current_time

        height, width = img.shape[:2]
        length = max(height, width)
        scale = length / opt.imgsz

        canvas = np.zeros((length, length, 3), dtype=np.uint8)
        canvas[:height, :width] = img
        blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (opt.imgsz, opt.imgsz), swapRB=True)

        outputs = session.run(None, {input_name: blob})[0]
        outputs = cv2.transpose(outputs[0])

        boxes = []
        scores = []
        for i in range(outputs.shape[0]):
            max_score = float(np.amax(outputs[i][4:]))
            if max_score >= opt.conf:
                x = (outputs[i][0] - 0.5 * outputs[i][2]) * scale
                y = (outputs[i][1] - 0.5 * outputs[i][3]) * scale
                w = outputs[i][2] * scale
                h = outputs[i][3] * scale
                boxes.append([x, y, w, h])
                scores.append(max_score)

        indices = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, opt.conf, opt.iou)
            if len(indices) > 0:
                indices = indices.flatten().tolist()

        color = (0, 255, 0)
        for idx in indices:
            x, y, w, h = map(int, boxes[idx])
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{scores[idx]:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 10, 41), 2)

        img_display = cv2.resize(img, (320, 320))
        cv2.imshow("Face Detection", img_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
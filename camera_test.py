import cv2

def open_camera():
    # 初始化摄像头，0表示默认摄像头（多个摄像头可依次尝试1、2...）
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接或权限！")
        return

    # 设置摄像头分辨率（可选，根据硬件支持调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 循环读取摄像头帧
    while True:
        # 读取一帧画面，ret为布尔值（是否读取成功），frame为帧数据
        ret, frame = cap.read()

        if not ret:
            print("无法读取摄像头画面，退出！")
            break

        # 实时显示画面
        cv2.imshow("Camera View", frame)

        # 按键控制：按q/ESC退出，按s保存当前帧
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27是ESC键ASCII码
            break
        elif key == ord('s'):
            cv2.imwrite("camera_capture.jpg", frame)
            print("画面已保存为camera_capture.jpg")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    open_camera()
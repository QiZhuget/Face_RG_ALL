import cv2
import os

# ===================== 配置参数 =====================
# 保存路径（直接使用你指定的路径）
SAVE_PATH = "images"
# 照片分辨率
IMG_WIDTH = 640
IMG_HEIGHT = 640
# ====================================================

# 创建保存文件夹（如果不存在）
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"已创建保存路径：{SAVE_PATH}")

# 打开摄像头（0=默认摄像头）
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    exit()

print("===== 拍照程序已启动 =====")
print("按 空格键 拍照")
print("按 Q 键 退出程序")

img_count = 0  # 照片计数

while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 显示实时画面
    cv2.imshow("Camera - Press SPACE to capture, Q to quit", frame)

    # 监听键盘按键
    key = cv2.waitKey(1) & 0xFF

    # 按空格键拍照
    if key == ord(' '):
        img_count += 1
        # 拼接保存路径和文件名
        save_file = os.path.join(SAVE_PATH, f"capture_{img_count}.jpg")
        # 保存照片
        cv2.imwrite(save_file, frame)
        print(f"✅ 已保存第 {img_count} 张照片：{save_file}")

    # 按Q键退出
    if key == ord('q'):
        print("程序已退出")
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
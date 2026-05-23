import cv2
import numpy as np
import time
import onnxruntime
import argparse

def preprocess_image(img_path, input_size=112, norm_range="0_1"):
    """
    预处理图片：BGR->RGB->resize->归一化->NCHW
    norm_range: "0_1" 或 "-1_1"
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"图片不存在或无法读取: {img_path}")

    # 转为 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 调整尺寸
    if img_rgb.shape[0] != input_size or img_rgb.shape[1] != input_size:
        img_rgb = cv2.resize(img_rgb, (input_size, input_size))

    # 归一化
    img_norm = img_rgb.astype(np.float32)
    if norm_range == "0_1":
        img_norm = img_norm / 255.0
    elif norm_range == "-1_1":
        img_norm = (img_norm - 127.5) / 128.0
    else:
        raise ValueError("norm_range 只能是 '0_1' 或 '-1_1'")

    # NCHW 格式并添加 batch 维度
    img_blob = np.transpose(img_norm, (2, 0, 1))  # CHW
    img_blob = np.expand_dims(img_blob, axis=0)  # 1CHW
    return img_blob


def main():
    parser = argparse.ArgumentParser(description="提取人脸特征并保存为npy文件")
    parser.add_argument("--model", default="../../weights/model_mobilefacenet_int8.onnx", help="INT8 模型路径")
    parser.add_argument("--image", default="images/face_8_1.jpg", help="输入人脸图片路径")
    parser.add_argument("--save_path", default="face_feature.npy", help="特征保存路径 (.npy)")
    parser.add_argument("--input_size", type=int, default=112, help="模型输入尺寸")
    parser.add_argument("--norm_range", default="0_1", choices=["0_1", "-1_1"], help="归一化范围")
    args = parser.parse_args()

    # 加载模型
    print(f"加载模型: {args.model}")
    sess = onnxruntime.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # 预处理
    input_tensor = preprocess_image(args.image, args.input_size, args.norm_range)

    # 推理
    start = time.perf_counter()
    outputs = sess.run([output_name], {input_name: input_tensor})
    end = time.perf_counter()

    # 特征
    feat = outputs[0][0]
    np.save(args.save_path, feat)

    # 输出
    print(f"\n✅ 特征提取完成！")
    print(f"耗时: {(end-start)*1000:.2f} ms")
    print(f"特征已保存到: {args.save_path}")
    print(f"特征维度: {feat.shape}")


if __name__ == "__main__":
    main()
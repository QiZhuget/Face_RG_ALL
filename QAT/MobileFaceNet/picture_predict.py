import cv2
import numpy as np
import time
import onnxruntime
import argparse


def preprocess_image(img_path, input_size=112, norm_range="0_1"):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"图片不存在: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (input_size, input_size))
    img_norm = img_rgb.astype(np.float32)
    if norm_range == "0_1":
        img_norm /= 255.0
    elif norm_range == "-1_1":
        img_norm = (img_norm - 127.5) / 128.0
    img_blob = np.transpose(img_norm, (2, 0, 1))
    img_blob = np.expand_dims(img_blob, axis=0)
    return img_blob


def cosine_similarity(a, b):
    """计算余弦相似度"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)


def main():
    parser = argparse.ArgumentParser(description="对比人脸特征相似度")
    parser.add_argument("--model", default="../../weights/model_mobilefacenet_int8.onnx", help="模型路径")
    parser.add_argument("--image", default="images/face_10_1.jpg", help="待测试图片路径")
    parser.add_argument("--feature_npy", default="face_feature.npy", help="已保存的特征文件路径 (.npy)")
    parser.add_argument("--input_size", type=int, default=112)
    parser.add_argument("--norm_range", default="0_1", choices=["0_1", "-1_1"])
    args = parser.parse_args()

    # 加载模型
    sess = onnxruntime.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # 读取已保存的特征
    feat1 = np.load(args.feature_npy)
    print(f"✅ 加载已保存特征: {args.feature_npy}")

    # 处理新图片
    input_tensor = preprocess_image(args.image, args.input_size, args.norm_range)
    outputs = sess.run([output_name], {input_name: input_tensor})
    feat2 = outputs[0][0]

    # 计算相似度
    sim = cosine_similarity(feat1, feat2)

    # 输出结果
    print("\n" + "="*50)
    print(f"人脸相似度分数: {sim:.4f}")
    if sim > 0.7:
        print("✅ 判定：同一人")
    else:
        print("❌ 判定：不是同一人")
    print("="*50)


if __name__ == "__main__":
    main()
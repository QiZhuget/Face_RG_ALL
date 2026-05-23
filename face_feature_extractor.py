import argparse
import cv2
import numpy as np
import glob
import os
from pathlib import Path
import Function

def save_face_features(face_database, save_path='output/face_features.npz'):
    """保存特征库到npz文件"""
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
    """从图片文件夹批量提取特征并保存"""
    # 1. 初始化模型
    providers = ['CPUExecutionProvider']
    if device != 'cpu' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"初始化 MobileFaceNet 模型（设备：{device}）...")
    mfn = Function.MobileFaceNetONNX(onnx_path=onnx_path, providers=providers)

    # 2. 检查图片文件夹
    if not os.path.exists(face_db_dir):
        os.makedirs(face_db_dir)
        print(f"人脸文件夹不存在，已创建：{face_db_dir}")
        print("   请将参考人脸图片放入该文件夹，命名格式：姓名.jpg（如 张三.jpg）")
        return

    # 3. 遍历图片提取特征
    face_database = {}
    valid_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    img_paths = glob.glob(os.path.join(face_db_dir, '*'))

    if not img_paths:
        print(f"{face_db_dir} 文件夹下无图片文件")
        return

    print(f"\n开始提取 {len(img_paths)} 个文件的特征...")
    for img_path in img_paths:
        # 过滤非图片文件
        if Path(img_path).suffix.lower() not in valid_ext:
            continue

        # 提取姓名（文件名去掉后缀）
        name = Path(img_path).stem
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过：无法读取 {img_path}")
            continue

        # 提取并归一化特征
        feat = mfn.get_feature(img)
        if feat is None:
            print(f"跳过：{name} 特征提取失败")
            continue

        feat = feat / (np.linalg.norm(feat) + 1e-12)
        face_database[name] = feat
        print(f"成功：{name} ({Path(img_path).name})")

    # 4. 保存特征库
    save_face_features(face_database, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='离线提取人脸特征并保存')
    parser.add_argument('--face_db_dir', default='face_database', help='人脸图片文件夹路径')
    parser.add_argument('--onnx_path', default='weights/model_mobilefacenet.onnx', help='MobileFaceNet ONNX 路径')
    parser.add_argument('--save_path', default='output/face_features.npz', help='特征库保存路径')
    parser.add_argument('--device', default='cpu', help='设备 (cpu/0)')
    opt = parser.parse_args()

    try:
        extract_features_from_folder(
            face_db_dir=opt.face_db_dir,
            onnx_path=opt.onnx_path,
            save_path=opt.save_path,
            device=opt.device
        )
    except Exception as e:
        print(f"程序出错：{e}")
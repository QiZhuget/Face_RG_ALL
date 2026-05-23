import os
import numpy as np
import cv2
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from onnxruntime.quantization import CalibrationMethod

"""
出现下面这个报错，是onnxruntime的问题，就换个环境或者重新装，这个环境下好像有点问题
Traceback (most recent call last):
  File "F:\0DU\YandM\QAT\Yolo\QAT.py", line 5, in <module>
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
  File "D:\Anaconda32021.11\envs\houmf\lib\site-packages\onnxruntime\quantization\__init__.py", line 1, in <module>
    from .calibrate import (  # noqa: F401
  File "D:\Anaconda32021.11\envs\houmf\lib\site-packages\onnxruntime\quantization\calibrate.py", line 22, in <module>
    from .quant_utils import apply_plot, load_model_with_shape_infer, smooth_distribution
  File "D:\Anaconda32021.11\envs\houmf\lib\site-packages\onnxruntime\quantization\quant_utils.py", line 145, in <module>
    onnx_proto.TensorProto.INT4: int4,  # base_dtype is np.int8
AttributeError: INT4
"""




class YOLOv8FaceDataReader(CalibrationDataReader):
    """YOLOv8人脸检测模型的校准数据读取器，用于INT8静态量化"""
    def __init__(self, calibration_image_folder: str, input_name: str = "images", input_size: int = 640):
        """
        参数:
            calibration_image_folder: 存放校准图片的文件夹路径
            input_name: ONNX模型的输入节点名称（通常是 "images"）
            input_size: 模型要求的输入边长（正方形）
        """
        self.input_name = input_name
        self.input_size = input_size
        self.enum_data_dicts = []
        self.datasize = 0

        # 检查文件夹是否存在
        if not os.path.exists(calibration_image_folder):
            raise FileNotFoundError(f"校准图片文件夹不存在: {calibration_image_folder}")

        # 获取文件夹下所有图片
        image_files = [os.path.join(calibration_image_folder, f) for f in os.listdir(calibration_image_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(image_files) == 0:
            raise ValueError(f"校准文件夹中没有找到任何jpg/jpeg/png图片: {calibration_image_folder}")

        print(f"找到 {len(image_files)} 张校准图片，开始预处理...")

        for idx, img_path in enumerate(image_files):
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图片 {img_path}，已跳过")
                continue

            # 1. 中心化缩放（保持宽高比，填充至正方形）
            h, w = img.shape[:2]
            scale = self.input_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
            canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            canvas[0:new_h, 0:new_w] = img_resized

            # 2. 归一化到 [0,1] 并转换格式 (HWC -> CHW, 添加 batch 维度)
            img_blob = canvas.astype(np.float32) / 255.0
            img_blob = np.transpose(img_blob, (2, 0, 1))      # CHW
            img_blob = np.expand_dims(img_blob, axis=0)       # 1CHW

            self.enum_data_dicts.append({self.input_name: img_blob})

        self.datasize = len(self.enum_data_dicts)
        print(f"成功加载 {self.datasize} 张有效校准图片。")

    def get_next(self):
        """CalibrationDataReader 要求实现的方法，返回下一个数据字典或 None"""
        if self.datasize <= 0:
            return None
        # 注意：pop() 会修改列表，每次调用返回一个样本
        if self.enum_data_dicts:
            return self.enum_data_dicts.pop()
        else:
            return None


if __name__ == "__main__":
    # ========== 用户配置区 ==========
    input_onnx_path = "../../weights/yolov8n-face-lindevs.onnx"   # 原始FP32模型路径
    output_onnx_path = "../../weights/yolov8n-face-lindevs-int8.onnx"  # 输出INT8模型路径
    calibration_folder = "images"               # 校准图片文件夹（需提前创建并放入图片）

    # 如果你的模型输入名不是 "images"，请先通过以下代码查看：
    # model = onnx.load(input_onnx_path)
    # print("模型输入名称:", model.graph.input[0].name)
    input_name = "images"   # 常见YOLOv8模型的输入名称，请根据实际情况修改
    # ================================

    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    # 检查原始模型是否存在
    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"找不到原始模型文件: {input_onnx_path}")

    # 创建校准数据读取器
    print("🚀 开始准备校准数据...")
    dr = YOLOv8FaceDataReader(calibration_folder, input_name=input_name, input_size=640)

    # 执行INT8静态量化
    print("🔧 正在执行INT8静态量化（这可能需要几分钟）...")
    quantize_static(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
        calibration_data_reader=dr,
        weight_type=QuantType.QInt8,          # 权重量化为INT8
        activation_type=QuantType.QInt8,      # 激活值量化为INT8
        quant_format=QuantFormat.QDQ,         # 量化格式：QDQ (Quantize-DeQuantize)
        op_types_to_quantize=['Conv', 'MatMul'],  # 只量化卷积和全连接层，保持精度
        calibrate_method=CalibrationMethod.Entropy,  # 校准算法：熵最小化
        extra_options={
            'ActivationSymmetric': True,       # 激活值对称量化
            'WeightSymmetric': True,           # 权重对称量化
            'ForceReduceRange': False
        }
    )
    print(f"✅ 量化完成！INT8模型已保存至: {output_onnx_path}")

    # 可选：对比模型大小
    size_fp32 = os.path.getsize(input_onnx_path) / (1024 * 1024)
    size_int8 = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"📊 模型大小对比: FP32 = {size_fp32:.2f} MB, INT8 = {size_int8:.2f} MB, 压缩率 = {size_int8/size_fp32:.1%}")
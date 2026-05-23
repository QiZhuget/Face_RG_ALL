import os
import cv2
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod

class MobileFaceNetDataReader(CalibrationDataReader):
    """适用于 MobileFaceNet 的校准数据读取器"""
    def __init__(self, calibration_image_folder: str, input_name: str = "images", input_size: int = 112):
        self.input_name = input_name
        self.input_size = input_size
        self.enum_data_dicts = []
        self.datasize = 0

        if not os.path.exists(calibration_image_folder):
            raise FileNotFoundError(f"校准文件夹不存在: {calibration_image_folder}")

        # 获取所有图片文件
        image_files = [f for f in os.listdir(calibration_image_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(image_files) == 0:
            raise ValueError(f"校准文件夹中没有找到任何 jpg/jpeg/png 图片: {calibration_image_folder}")

        print(f"找到 {len(image_files)} 张校准图片，开始预处理...")

        for img_file in image_files:
            img_path = os.path.join(calibration_image_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取 {img_path}，已跳过")
                continue

            # 确保图片尺寸为 112x112（如果已经是则跳过 resize）
            if img.shape[0] != input_size or img.shape[1] != input_size:
                img = cv2.resize(img, (input_size, input_size))

            # 转换色彩空间 BGR -> RGB（模型训练时通常使用 RGB）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 归一化到 [0, 1]（常见做法，如果训练时使用 [-1,1] 请修改下一行）
            img_norm = img_rgb.astype(np.float32) / 255.0

            # 转换为 NCHW 格式并增加 batch 维度
            img_blob = np.transpose(img_norm, (2, 0, 1))   # CHW
            img_blob = np.expand_dims(img_blob, axis=0)    # 1CHW

            self.enum_data_dicts.append({self.input_name: img_blob})

        self.datasize = len(self.enum_data_dicts)
        print(f"成功加载 {self.datasize} 张有效校准图片")

    def get_next(self):
        """返回下一个校准数据，迭代完成后返回 None"""
        if self.datasize <= 0 or not self.enum_data_dicts:
            return None
        return self.enum_data_dicts.pop()


if __name__ == "__main__":
    # ========== 配置参数（请根据实际情况修改）==========
    input_onnx_path = "../../weights/model_mobilefacenet.onnx"        # 原始 FP32 模型
    output_onnx_path = "../../weights/model_mobilefacenet_int8.onnx"  # 输出 INT8 模型
    calibration_folder = "images"    # 校准图片文件夹

    # 模型输入名称（通过 Netron 或 onnxruntime 查看得到）
    input_name = "images"

    # 可选：如果训练时使用的是 [-1,1] 归一化，将下面变量改为 True
    use_norm_minus1_1 = False   # 默认使用 [0,1] 归一化
    # =================================================

    # 创建 DataReader 时动态选择归一化方式（如需修改，可以在类内部调整）
    # 若有需要，可以直接修改类中的 img_norm 计算行，或通过参数传递。
    # 这里保持简单，直接使用 [0,1] 归一化。

    # 检查输入模型是否存在
    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"找不到原始模型文件: {input_onnx_path}")

    # 创建校准数据读取器
    print("🚀 开始准备校准数据...")
    dr = MobileFaceNetDataReader(calibration_folder, input_name=input_name, input_size=112)

    # 执行 INT8 静态量化
    print("🔧 正在执行 INT8 静态量化（可能需要几分钟）...")
    quantize_static(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
        calibration_data_reader=dr,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=['Conv', 'MatMul'],   # 量化卷积和全连接层
        calibrate_method=CalibrationMethod.Entropy, # 使用熵校准
        extra_options={
            'ActivationSymmetric': True,
            'WeightSymmetric': True,
            'ForceReduceRange': False
        }
    )

    print(f"✅ 量化完成！INT8 模型已保存至: {output_onnx_path}")

    # 对比模型大小
    size_fp32 = os.path.getsize(input_onnx_path) / (1024 * 1024)
    size_int8 = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"📊 模型大小对比: FP32 = {size_fp32:.2f} MB, INT8 = {size_int8:.2f} MB, 压缩率 = {size_int8/size_fp32:.1%}")

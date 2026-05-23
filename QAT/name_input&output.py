import onnxruntime
import pprint

# 1. 指定你的MobileFaceNet ONNX模型路径
onnx_path = "../../weights/model_mobilefacenet.onnx"

# 2. 创建一个ONNX Runtime会话（InferenceSession）
session = onnxruntime.InferenceSession(onnx_path)

# 3. 获取并打印出模型的所有输入信息
print("----------------- 模型输入部分 -----------------")
for input_tensor in session.get_inputs():
    input_info = {
        "name": input_tensor.name,
        "type": input_tensor.type,
        "shape": input_tensor.shape,
    }
    pprint.pprint(input_info)

# 4. （可选）再查看一下模型的输出信息，方便后续使用
print("\n----------------- 模型输出部分 -----------------")
for output_tensor in session.get_outputs():
    output_info = {
        "name": output_tensor.name,
        "type": output_tensor.type,
        "shape": output_tensor.shape,
    }
    pprint.pprint(output_info)
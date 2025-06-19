import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# 常量配置
MODEL_PATH = "model.tflite"      # 模型路径
LABELS_PATH = "labels.txt"       # 标签路径
IMAGE_DIR = "images"             # 图片目录

# 加载标签
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f.readlines()]

# 初始化TFLite解释器
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 获取模型输入输出详情
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # 输入尺寸（如 [224, 224]）

# 处理单张图片
def predict_image(image_path):
    # 读取并预处理图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return None
    
    # 调整尺寸、归一化（根据模型要求修改）
    img_resized = cv2.resize(img, (input_shape[1], input_shape[0]))
    input_data = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    # 执行推理
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 解析结果
    predicted_class = np.argmax(output_data[0])
    confidence = output_data[0][predicted_class]
    return labels[predicted_class], float(confidence)

# 遍历图片目录并识别
for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(IMAGE_DIR, filename)
        class_name, confidence = predict_image(image_path)
        print(f"图片: {filename} → 预测类别: {class_name} (置信度: {confidence:.2f})")
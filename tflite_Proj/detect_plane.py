import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# 加载模型和标签
model_path = "model.tflite"
labels_path = "labels.txt"

# 初始化 TFLite 解释器
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 获取输入输出张量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 加载标签
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理图像（调整大小、归一化等）
    input_data = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    input_data = input_data / 255.0  # 归一化（根据模型要求调整）

    # 执行推理
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 解析结果
    predicted_class = np.argmax(output_data[0])
    confidence = output_data[0][predicted_class]
    label = f"{labels[predicted_class]}: {confidence:.2f}"

    # 在图像上显示结果
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-time Detection", frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
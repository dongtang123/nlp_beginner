import cv2
import numpy as np

# 读取图像
image = cv2.imread('51078120_01.png')
orig = image.copy()

# 转换图像为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 加载 EAST 文本检测模型
net = cv2.dnn.readNet(r'D:\nlp\frozen_east_text_detection.pb')

# 从图像中获取输入尺寸
height, width = image.shape[:2]
new_height = (height // 32) * 32
new_width = (width // 32) * 32

# 缩放图像
blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), True, False)
net.setInput(blob)

# 获取文本检测结果
(scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

# 获取文本框的几何信息和分数
rectangles, confidences = [], []
for i in range(scores.shape[2]):
    # 忽略低置信度的框
    if scores[0, 0, i, 0] < 1:  # 修改此处
        continue

    # 计算文本框的角度和尺寸
    angle = geometry[0, 0, i, 4] * 90
    cos = np.cos(np.deg2rad(angle))
    sin = np.sin(np.deg2rad(angle))
    h = geometry[0, 0, i, 0] + geometry[0, 0, i, 2]
    w = geometry[0, 0, i, 1] + geometry[0, 0, i, 3]

    # 计算文本框的中心点坐标
    x_center = geometry[0, 0, i, 5] * new_width
    y_center = geometry[0, 0, i, 6] * new_height

    # 计算文本框的四个顶点坐标
    x0 = x_center - w * cos - h * sin
    y0 = y_center - w * sin + h * cos
    x1 = x_center + w * cos - h * sin
    y1 = y_center - w * sin - h * cos
    x2 = x_center + w * cos + h * sin
    y2 = y_center + w * sin - h * cos
    x3 = x_center - w * cos + h * sin
    y3 = y_center + w * sin + h * cos

    # 添加文本框的四个顶点坐标
    rectangles.append((x0, y0, x1, y1, x2, y2, x3, y3))
    confidences.append(scores[0, 0, i])

# 将 rectangles 转换为 numpy 数组
rectangles = np.array(rectangles)

# 应用非极大值抑制来移除重叠的文本框
indices = cv2.dnn.NMSBoxesRotated(rectangles, confidences, 0.5, 0.5)

# 在原始图像上绘制保留的文本框
for i in indices:
    # 获取文本框的四个顶点坐标
    rect = rectangles[i[0]]
    box = np.array([[rect[0], rect[1]], [rect[2], rect[3]], [rect[4], rect[5]], [rect[6], rect[7]]], dtype=np.int32)

    # 绘制文本框
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

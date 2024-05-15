import cv2
import numpy as np


input_image_path = "Snipaste_2024-01-13_17-02-20.png"
output_image_path = "binary_test.png"


def remove_lines(image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path, 0)
    # 使用阈值二值化
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # 使用形态学操作找到横线
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # 从原始图像中减去检测到的线条
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = cv2.morphologyEx(detected_lines, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    final_image = cv2.bitwise_xor(img, result)

    # 保存处理后的图像
    cv2.imwrite(output_path, final_image)

# 使用函数
# remove_lines(input_image_path, output_image_path)

def extract_letters(image_path, output_folder):
    # 读取图像并转换为灰度图
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对于每个轮廓，裁剪并保存文字
    for i, contour in enumerate(contours):
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 可能需要根据实际情况调整这里的条件，以确保正确地识别文字
        if w > 8 and h > 8:  # 轮廓大小过滤
            # 裁剪图像
            letter_image = img[y:y + h, x:x + w]

            # 保存裁剪的图像
            cv2.imwrite(f"{output_folder}/letter_{i}.png", letter_image)

# 使用函数
extract_letters(output_image_path, 'process_image')



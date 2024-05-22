import os
import cv2
import numpy as np

def add_red_box(image, top_left, bottom_right, thickness=2):
    return cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), thickness)

def enlarge_and_display(image, top_left, bottom_right, scale_factor):
    enlarged_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    enlarged_region = cv2.resize(enlarged_region, None, fx=scale_factor, fy=scale_factor)
    h, w, _ = image.shape
    offset_x = w - enlarged_region.shape[1]
    offset_y = h - enlarged_region.shape[0]
    # image[offset_y:h, offset_x :w] = enlarged_region  #  放在右下角
    image[offset_y:h, 0:enlarged_region.shape[1]] = enlarged_region  # 放在左下角
    return image

def process_images(input_folder, output_folder, top_left, bottom_right, scale_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to read image: {input_path}")
                continue

            image_with_box = add_red_box(image.copy(), top_left, bottom_right)
            final_image = enlarge_and_display(image_with_box, top_left, bottom_right, scale_factor)

            cv2.imwrite(output_path, final_image)
            print(f"Processed image: {output_path}")

if __name__ == "__main__":
    input_folder = r"C:\Users\hu\Desktop\2\xuan\tno\4"   # 输入图像文件夹路径
    output_folder = r"C:\Users\hu\Desktop\2\xuan\tno\4-1" # 输出处理后的图像保存路径
    top_left = (375,360)                 # 指定区域的左上角坐标 (x, y)
    bottom_right = (560,480)             # 指定区域的右下角坐标 (x, y)
    scale_factor = 1.8                # 放大倍数

    process_images(input_folder, output_folder, top_left, bottom_right, scale_factor)

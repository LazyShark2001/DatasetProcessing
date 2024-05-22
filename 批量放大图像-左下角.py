import os
import cv2
import numpy as np

def add_red_box(image, top_left, bottom_right, thickness=2):
    return cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), thickness)

def enlarge_and_display(image, bottom_left, top_right, scale_factor):
    enlarged_region = image[bottom_left[1]:top_right[1], bottom_left[0]:top_right[0]]
    enlarged_region = cv2.resize(enlarged_region, None, fx=scale_factor, fy=scale_factor)
    h, w, _ = image.shape
    offset_x = w - enlarged_region.shape[1]
    image[h-enlarged_region.shape[0]:h, offset_x:w] = enlarged_region
    return image

def process_images(input_folder, output_folder, bottom_left, top_right, scale_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to read image: {input_path}")
                continue

            image_with_box = add_red_box(image.copy(), bottom_left, top_right)
            final_image = enlarge_and_display(image_with_box, bottom_left, top_right, scale_factor)

            cv2.imwrite(output_path, final_image)
            print(f"Processed image: {output_path}")

if __name__ == "__main__":
    input_folder = r"C:\Users\hu\Desktop\road-1"   # 输入图像文件夹路径
    output_folder = r"C:\Users\hu\Desktop\road-1-1" # 输出处理后的图像保存路径
    bottom_left = (165, 140)              # 指定区域的左下角坐标 (x, y)
    top_right = (200, 180)                # 指定区域的右上角坐标 (x, y)
    scale_factor = 2.0                    # 放大倍数

    process_images(input_folder, output_folder, bottom_left, top_right, scale_factor)

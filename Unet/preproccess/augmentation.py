import os
import cv2
import numpy as np
import random
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

def elastic_transform(image, alpha, sigma, alpha_affine=0, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_."""
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    if len(shape) == 3:
        distorted_image = np.zeros_like(image)
        for i in range(shape[2]):
            distorted_image[..., i] = map_coordinates(image[..., i], indices, order=1, mode='reflect').reshape(shape_size)
    else:
        distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape_size)
    
    return distorted_image

def rotate_image(image, angle):
    """Rotate image by given angle"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    return rotated_image

def flip_image(image, flip_code):
    """Flip image horizontally or vertically"""
    return cv2.flip(image, flip_code)

def augment_data(input_data_dir, input_ground_dir, output_data_dir, output_ground_dir, num_augmentations=1):
    """对数据进行增强处理"""
    
    # 创建输出目录
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_ground_dir, exist_ok=True)
    
    # 获取所有原图文件名
    data_files = [f for f in os.listdir(input_data_dir) if f.endswith('.png')]
    
    for file_name in data_files:
        # 读取原图和ground truth
        data_path = os.path.join(input_data_dir, file_name)
        ground_name = file_name  # 假设文件名一致
        ground_path = os.path.join(input_ground_dir, ground_name)
        
        if not os.path.exists(ground_path):
            print(f"Ground truth for {file_name} not found, skipping...")
            continue
            
        data_image = cv2.imread(data_path)
        ground_image = cv2.imread(ground_path)
        
        # 保存原始图像
        cv2.imwrite(os.path.join(output_data_dir, file_name), data_image)
        cv2.imwrite(os.path.join(output_ground_dir, ground_name), ground_image)
        
        # 进行数据增强
        # 随机选择一种增强方式
        aug_type = random.randint(0, 2)
        
        # 生成增强后的文件名
        name, ext = os.path.splitext(file_name)
        augmented_name = f"{name}_aug{ext}"
        
        ground_name, ground_ext = os.path.splitext(ground_name)
        augmented_ground_name = f"{ground_name}_aug{ground_ext}"
        
        augmented_data = None
        augmented_ground = None
        
        if aug_type == 0:  # 旋转
            angle = random.uniform(-30, 30)  # 随机角度-30到30度
            augmented_data = rotate_image(data_image, angle)
            augmented_ground = rotate_image(ground_image, angle)
        elif aug_type == 1:  # 翻转
            flip_code = random.choice([-1, 0, 1])  # -1: 同时水平垂直翻转, 0: 垂直翻转, 1: 水平翻转
            augmented_data = flip_image(data_image, flip_code)
            augmented_ground = flip_image(ground_image, flip_code)
        elif aug_type == 2:  # 弹性变形
            alpha = data_image.shape[1] * 2
            sigma = data_image.shape[1] * 0.08
            augmented_data = elastic_transform(data_image, alpha=alpha, sigma=sigma)
            augmented_ground = elastic_transform(ground_image, alpha=alpha, sigma=sigma)
        
        # 保存增强后的图像
        cv2.imwrite(os.path.join(output_data_dir, augmented_name), augmented_data)
        cv2.imwrite(os.path.join(output_ground_dir, augmented_ground_name), augmented_ground)

if __name__ == "__main__":
    # 定义路径
    train_data_dir = "../dataset/train/Data_png"
    train_ground_dir = "../dataset/train/Ground/Liver"
    aug_data_dir = "../dataset/augmentation/Data_png"
    aug_ground_dir = "../dataset/augmentation/Ground"
    
    # 执行数据增强
    augment_data(train_data_dir, train_ground_dir, aug_data_dir, aug_ground_dir, num_augmentations=1)
    
    print("数据增强完成！")

# -*- coding: utf-8 -*-
import cv2
import os

# 设置待处理的文件夹路径
folder_path = '/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/example-data/'

# 获取文件夹下所有jpg文件的文件名列表
file_list = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 处理每个jpg文件
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    
    # 读取图像
    img = cv2.imread(file_path)
    
    # 如果成功读取图像
    if img is not None:
        # 将图像全部设置为黑色（全零）
        img[:] = 0
        
        # 保存全黑的图像为同名的jpg文件
        new_file_path = os.path.join(folder_path, file_name)  # 保存为新文件名
        cv2.imwrite(new_file_path, img)
        # print(f"Processed: {file_name}")
    # else:
        # print(f"Failed to process: {file_name}")
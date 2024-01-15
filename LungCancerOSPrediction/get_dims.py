import glob

import nibabel as nib
import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# # 获取目录中所有NIfTI文件的路径
# def get_nii_files(directory):
#     return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii')]
#
# # 读取NIfTI文件并获取其尺寸
# def get_dimensions(file_path):
#     nii_data = nib.load(file_path)
#     print(file_path,nii_data.shape)
#     return nii_data.shape
#
# # 绘制直方图
# def plot_dimension_histograms(dimensions):
#     dimensions = np.array(dimensions)
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#     # 维度0
#     axes[0].hist(dimensions[:, 0], bins=50, color='red')
#     axes[0].set_title('Dimension 0 Size Histogram')
#
#     # 维度1
#     axes[1].hist(dimensions[:, 1], bins=50, color='green')
#     axes[1].set_title('Dimension 1 Size Histogram')
#
#     # 维度2
#     axes[2].hist(dimensions[:, 2], bins=50, color='blue')
#     axes[2].set_title('Dimension 2 Size Histogram')
#
#     plt.tight_layout()
#     plt.show()
#
# # 您的NIfTI文件所在的目录
# # directory_path = 'path_to_your_nii_files_directory'
# image_files = glob.glob('/media/ubuntu/disk/dataset/MEDdataset/chaimeleon/*/*/*.nii.gz')
#
# # 获取所有文件的尺寸
# dimensions = [get_dimensions(file) for file in image_files]
#
# # 计算平均尺寸
# average_dimensions = np.mean(dimensions, axis=0)
# print("平均尺寸：", average_dimensions)
#
# # 绘制直方图
# plot_dimension_histograms(dimensions)
# import nibabel as nib
# import numpy as np
# import os
#
#
# # 加载NIfTI图像
# def load_nii(file_path):
#     return nib.load(file_path)
#
#
# # 剪裁并保存NIfTI图像
# def crop_and_save_nii(nii_image, save_path, new_depth):
#     # 获取图像数据和仿射矩阵
#     image_data = nii_image.get_fdata()
#     affine = nii_image.affine
#
#     # 计算新的起始和结束索引
#     depth = image_data.shape[2]
#     start = (depth - new_depth) // 2
#     end = start + new_depth
#
#     # 剪裁图像
#     cropped_image_data = image_data[start:end, start:end,start:end]
#
#     # 创建新的NIfTI图像
#     cropped_nii = nib.Nifti1Image(cropped_image_data, affine)
#
#     # 保存新的NIfTI图像
#     nib.save(cropped_nii, save_path)
#
#
# # 您的NIfTI文件路径
# file_path = '/media/ubuntu/disk/dataset/MEDdataset/chaimeleon/cfio5 (1)/case_0094/case_0094.nii.gz'
#
# # 剪裁后的文件保存路径
# save_path = 'cropped_image.nii'
#
# # 要剪裁到的新深度
# new_depth = 384
#
# # 加载NIfTI图像
# nii_image = load_nii(file_path)
#
# # 剪裁并保存图像
# crop_and_save_nii(nii_image, save_path, new_depth)


import json
import os

# 定义要搜索的目录路径
# directory_path = '/path/to/your/json_files_directory'
image_files = glob.glob('/media/ubuntu/disk/dataset/MEDdataset/chaimeleon/*/*/*ground_truth.json')

# 初始化存储生存时间的列表
survival_times = []

# 遍历目录中的所有文件
for filename in image_files:
    if filename.endswith('.json'):
        # 构造文件的完整路径
        # file_path = os.path.join(directory_path, filename)

        # 打开并读取JSON文件
        with open(filename, 'r') as file:
            data = json.load(file)

            # 提取survival_time_months键的值并添加到列表中
            if 'survival_time_months' in data:
                survival_times.append(data['survival_time_months'])

# 计算均值
if survival_times:
    average_survival_time = sum(survival_times) / len(survival_times)
    print("Average survival time (months):", average_survival_time)
else:
    print("No survival time data found.")


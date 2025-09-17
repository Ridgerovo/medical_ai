import pydicom
import os
import matplotlib.pyplot as plt
from skimage import img_as_float

# 定义三个数据集路径
path_train = "../dataset/train/Data"
path_val = "../dataset/val/Data"
path_test = "../dataset/test/Data"

# 定义输出路径
output_train = "../dataset/train/Data_png"
output_val = "../dataset/val/Data_png"
output_test = "../dataset/test/Data_png"


def dicom_2png(orifile, savefile, width, height):
    _currFile = orifile
    dcm = pydicom.dcmread(orifile)
    # fileName = os.path.basename(file)
    imageX = dcm.pixel_array
    temp = imageX.copy()
    picMax = imageX.max()
    vmin = imageX.min()
    vmax = temp[temp < picMax].max()
    # print("vmin : ", vmin)
    # print("vmax : ", vmax)
    imageX[imageX > vmax] = 0
    imageX[imageX < vmin] = 0
    # result = exposure.is_low_contrast(imageX)
    # # print(result)
    image = img_as_float(imageX)
    plt.cla()
    plt.figure('adjust_gamma', figsize=(width / 100, height / 100))
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.imshow(image, 'gray')
    plt.axis('off')
    plt.savefig(savefile)


if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_val, exist_ok=True)
    os.makedirs(output_test, exist_ok=True)
    
    # 处理训练集
    if os.path.exists(path_train):
        names = os.listdir(path_train)
        for i in range(len(names)):
            dicom_path = os.path.join(path_train, names[i])
            png_name = os.path.splitext(names[i])[0]
            dst_path = os.path.join(output_train, (png_name + '.png'))
            dicom_2png(dicom_path, dst_path, 256, 256)  # 修改尺寸为256x256

    # 处理验证集
    if os.path.exists(path_val):
        names = os.listdir(path_val)
        for i in range(len(names)):
            dicom_path = os.path.join(path_val, names[i])
            png_name = os.path.splitext(names[i])[0]
            dst_path = os.path.join(output_val, (png_name + '.png'))
            dicom_2png(dicom_path, dst_path, 256, 256)  # 修改尺寸为256x256

    # 处理测试集
    if os.path.exists(path_test):
        names = os.listdir(path_test)
        for i in range(len(names)):
            dicom_path = os.path.join(path_test, names[i])
            png_name = os.path.splitext(names[i])[0]
            dst_path = os.path.join(output_test, (png_name + '.png'))
            dicom_2png(dicom_path, dst_path, 256, 256)  # 修改尺寸为256x256
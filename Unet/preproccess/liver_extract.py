import os
import cv2


def makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def extract_liver(dataset_dir):
    src_names = os.listdir(dataset_dir)
    if src_names[0] == 'Liver':
        src_names.remove('Liver')
    src_count = len(src_names)
    dst_dir = os.path.join(dataset_dir, "Liver")
    makedir(dst_dir)
    for num in range(src_count):
        src_path = os.path.join(dataset_dir, src_names[num])
        src = cv2.imread(src_path)   # OpenCV读进来要指定是灰度图像，不然默认三通道。这里之前忘记指定了
        # flag = 0
        flag = 1
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                for k in range(src.shape[2]):
                    if 55 <= src.item(i, j, k) <= 70:
                        flag = 1  # 表示有肝脏
                        src.itemset((i, j, k), 255)
                    else:
                        src.itemset((i, j, k), 0)
        if flag == 1:
            dst_path = os.path.join(dst_dir, src_names[num])
            cv2.imwrite(dst_path, src)


if __name__ == '__main__':
    train_dir = os.path.join("../dataset", "train", "Ground")
    # 修改测试集路径为相对路径，与数据集划分脚本保持一致
    test_dir = os.path.join("../dataset", "test", "Ground")
    # 添加验证集路径
    val_dir = os.path.join("../dataset", "val", "Ground")
    extract_liver(train_dir)
    extract_liver(test_dir)
    # 对验证集也进行肝脏提取操作
    extract_liver(val_dir)
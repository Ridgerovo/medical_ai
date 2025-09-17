from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root):
    imgs = []
    ori_path = os.path.join(root, "Data_png")
    # 修改groundtruth路径到指定目录
    ground_path = os.path.join(root, "Ground", "Liver")

    # 获取实际存在的文件列表
    data_names = set(os.listdir(ori_path))
    ground_names = set(os.listdir(ground_path))

    # 找到共同的文件名
    common_names = data_names.intersection(ground_names)

    # 只添加实际存在的配对文件
    for name in common_names:
        img_path = os.path.join(ori_path, name)
        mask_path = os.path.join(ground_path, name)
        # 验证文件是否可以正常打开
        try:
            # 尝试打开图像文件验证其有效性
            with Image.open(img_path) as img_data, Image.open(mask_path) as img_ground:
                # 确保图像可以正常加载
                img_data.load()
                img_ground.load()
            imgs.append((img_path, mask_path))
        except Exception as e:
            print(f"警告: 无法加载图像对 {name}: {e}")
            continue

    return imgs


class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('L')
        img_y = Image.open(y_path).convert('L')

        # 确保在应用transform前检查图像尺寸
        # 如果transform不为None，则应用transform
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
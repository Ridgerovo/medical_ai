import torch
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import ResNetUNet, CombinedLoss
from dataset import LiverDataset
from common_tools import transform_invert


def makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


val_interval = 1
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# 均为灰度图像，只需要转换为tensor
# Resize操作以将图像调整为256x256
x_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
y_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_curve = list()
valid_curve = list()


def get_current_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, criterion, optimizer, dataload, num_epochs=100):
    makedir('./model')
    model_path = f"./model/weights_300_resnet.pth"
    best_model_path = f"./model/best_weights_resnet.pth"

    # 初始化最佳验证损失为无穷大
    best_val_loss = float('inf')
    # 添加早停相关变量
    patience = 15  # 容忍验证损失不降低的epoch数
    patience_counter = 0  # 计数器

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        start_epoch = 20
        print('加载成功！')
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

    # 使用余弦退火学习率调度器，调整参数以适应更长的训练周期
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*2, eta_min=1e-7)

    for epoch in range(start_epoch + 1, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0

        # 显示当前学习率
        current_lr = get_current_lr(optimizer)
        print(f"learning rate: {current_lr:.6f}")

        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_curve.append(loss.item())
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))

        # 更新学习率
        scheduler.step()

        # 显示更新后的学习率（如果发生变化）
        new_lr = get_current_lr(optimizer)
        if new_lr != current_lr:
            print(f"Learning rate updated to: {new_lr:.6f}")

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f'./model/weights_{epoch + 1}_resnet.pth')

        # Validate the model
        # 修改: 使用验证集而不是测试集进行验证
        val_dataset = LiverDataset("dataset/val", transform=x_transforms, target_transform=y_transforms)
        # 添加验证集数据检查
        if len(val_dataset) == 0:
            print("警告: 验证数据集为空!")
        else:
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
            if (epoch + 2) % val_interval == 0:
                loss_val = 0.
                model.eval()
                with torch.no_grad():
                    step_val = 0
                    for x, y in val_loader:
                        step_val += 1
                        x = x.type(torch.FloatTensor)
                        inputs = x.to(device)  # 确保验证数据也移动到GPU
                        labels = y.to(device)  # 确保验证标签也移动到GPU
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss_val += loss.item()

                    avg_val_loss = loss_val / step_val
                    valid_curve.append(avg_val_loss)
                    print("epoch %d val_loss:%0.3f" % (epoch, avg_val_loss))

                    # 早停机制和模型保存
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0  # 重置耐心计数器
                        torch.save(model.state_dict(), best_model_path)
                        print(f"保存最佳模型，验证损失: {avg_val_loss:.3f}")
                    else:
                        patience_counter += 1
                        print(f"验证损失未改善，耐心计数: {patience_counter}/{patience}")
                        
                    # 如果验证损失连续patience个epoch没有改善，则早停
                    if patience_counter >= patience:
                        print(f"早停机制触发，在epoch {epoch} 停止训练")
                        # 绘制训练曲线并返回
                        if len(train_curve) > 0:
                            train_x = range(len(train_curve))
                            train_y = train_curve

                            train_iters = len(dataload)
                            valid_x = np.arange(1, len(
                                valid_curve) + 1) * train_iters * val_interval
                            valid_y = valid_curve

                            plt.plot(train_x, train_y, label='Train')
                            plt.plot(valid_x, valid_y, label='Val')

                            plt.legend(loc='upper right')
                            plt.ylabel('loss value')
                            plt.xlabel('Iteration')
                            plt.title('Training and Validation Loss')
                            plt.show()
                        return model
                model.train()  # 恢复训练模式

    # 只在有训练数据时绘制曲线
    if len(train_curve) > 0:
        train_x = range(len(train_curve))
        train_y = train_curve

        train_iters = len(dataload)
        valid_x = np.arange(1, len(
            valid_curve) + 1) * train_iters * val_interval  # 由于val中记录的是EpochLoss，需要对记录点进行转换到iterations
        valid_y = valid_curve

        plt.plot(train_x, train_y, label='Train')
        plt.plot(valid_x, valid_y, label='Val')

        plt.legend(loc='upper right')
        plt.ylabel('loss value')
        plt.xlabel('Iteration')
        plt.show()
    return model


# 训练模型
def train(args):
    # 使用ResNet作为编码器的UNet
    model = ResNetUNet(1, 1, pretrained=not args.no_pretrain).to(device)

    # 使用交叉熵损失+Dice系数优化
    criterion = CombinedLoss()

    batch_size = args.batch_size
    # 调整优化器参数，降低初始学习率
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    liver_dataset = LiverDataset("./dataset/augmentation", transform=x_transforms, target_transform=y_transforms)


    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders, num_epochs=150)


# 显示模型的输出结果
def predict(args):
    # 使用ResNet作为编码器的UNet
    model = ResNetUNet(1, 1, pretrained=False)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))  # 使用device加载模型
    model.to(device)  # 将模型移动到GPU

    # 添加GPU使用提示
    if next(model.parameters()).is_cuda:
        print("Model loaded on GPU")
    else:
        print("Model loaded on CPU")

    # 使用测试集进行预测
    liver_dataset = LiverDataset("./dataset/test", transform=x_transforms, target_transform=y_transforms)

    # 添加测试集大小检查
    print(f"测试数据集大小: {len(liver_dataset)}")
    if len(liver_dataset) == 0:
        print("错误: 测试数据集为空，请检查数据是否已正确预处理")
        return

    dataloaders = DataLoader(liver_dataset, batch_size=1)

    save_root = './dataset/predict'
    # 确保保存目录存在
    makedir(save_root)

    model.eval()
    plt.ion()
    index = 0
    with torch.no_grad():
        for x, ground in dataloaders:
            x = x.type(torch.FloatTensor)
            x = x.to(device)  # 将输入数据移动到GPU

            # 添加GPU使用提示
            if x.is_cuda:
                print(f"Processing image {index} on GPU")
            else:
                print(f"Processing image {index} on CPU")

            y = model(x)
            x = x.cpu()  # 将数据移回CPU进行后续处理
            y = y.cpu()  # 将结果移回CPU进行后续处理
            x = torch.squeeze(x)
            x = x.unsqueeze(0)
            ground = torch.squeeze(ground)
            ground = ground.unsqueeze(0)
            img_ground = transform_invert(ground, y_transforms)
            img_x = transform_invert(x, x_transforms)
            img_y = torch.sigmoid(torch.squeeze(y)).numpy()  # 应用sigmoid激活函数

            # 后处理：阈值处理
            img_y = (img_y > 0.5).astype(np.uint8) * 255
            img_y = cv2.resize(img_y, (256, 256), interpolation=cv2.INTER_NEAREST)

            # cv2.imshow('img', img_y)
            src_path = os.path.join(save_root, "predict_%d_s.png" % index)
            save_path = os.path.join(save_root, "predict_%d_o.png" % index)
            ground_path = os.path.join(save_root, "predict_%d_g.png" % index)
            img_ground.save(ground_path)
            # img_x.save(src_path)
            cv2.imwrite(save_path, img_y)
            index = index + 1
            # plt.imshow(img_y)
            # plt.pause(0.5)
        # plt.show()


# 计算Dice系数
def dice_calc(args):
    root = './dataset/predict'
    if not os.path.exists(root):
        print("预测结果目录不存在，请先运行预测")
        return

    nums = len([f for f in os.listdir(root) if f.endswith('_g.png')])
    if nums == 0:
        print("没有找到预测结果，请先运行预测")
        return

    dice = list()
    dice_mean = 0
    for i in range(nums):
        ground_path = os.path.join(root, "predict_%d_g.png" % i)
        predict_path = os.path.join(root, "predict_%d_o.png" % i)
        if not os.path.exists(ground_path) or not os.path.exists(predict_path):
            print(f"警告: 缺少第{i}组预测结果文件")
            continue

        img_ground = cv2.imread(ground_path)
        img_predict = cv2.imread(predict_path)

        # 添加尺寸检查
        if img_ground is None or img_predict is None:
            print(f"Warning: Could not read image {ground_path} or {predict_path}")
            continue

        # 获取图像的实际尺寸
        height, width = img_ground.shape[:2]
        pred_height, pred_width = img_predict.shape[:2]

        # 确保两个图像尺寸一致，将预测图像调整为与ground truth相同尺寸
        if height != pred_height or width != pred_width:
            print(f"Warning: Image sizes don't match for pair {i}, resizing predict image")
            img_predict = cv2.resize(img_predict, (width, height))

        intersec = 0
        x = 0
        y = 0
        # 使用实际图像尺寸进行计算
        for w in range(height):
            for h in range(width):
                intersec += img_ground.item(w, h, 1) * img_predict.item(w, h, 1) / (255 * 255)
                x += img_ground.item(w, h, 1) / 255
                y += img_predict.item(w, h, 1) / 255
        if x + y == 0:
            current_dice = 1
        else:
            current_dice = round(2 * intersec / (x + y), 3)
        dice_mean += current_dice
        dice.append(current_dice)
    if len(dice) > 0:
        dice_mean /= len(dice)
        print(dice)
        print(round(dice_mean, 3))
    else:
        print("没有有效的Dice系数可以计算")


if __name__ == '__main__':
    # 参数解析
    parse = argparse.ArgumentParser()
    # 修改: 将默认action从train改为predict，避免混淆
    parse.add_argument("--action", type=str, help="train, predict or dice", default="train")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file",
                       default="./model/best_weights_resnet.pth")
    # 添加no_pretrain参数
    parse.add_argument("--no_pretrain", action="store_true", help="disable pretrained weights for resnet")
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    # 修改: 将test改为predict
    elif args.action == "predict":
        predict(args)
    elif args.action == "dice":
        dice_calc(args)
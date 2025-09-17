#!/C:/Users/86157/miniconda3/Lib/medical_ai/Scripts/python.exe
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from unet import ResNetUNet  # 假设你有这个模型类
import os
import shutil
import argparse
import json
from PIL import Image
from datetime import datetime
import sys
import pydicom

# 配置固定参数
MODEL_PATH = r"C:\Users\86157\Desktop\实习\Unet\model\best_weights_resnet.pth"  # 模型路径
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(CURRENT_SCRIPT_DIR, "segmentation_results")

# 设置matplotlib非交互后端
plt.switch_backend('Agg')
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 图像转换（与训练一致）
x_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def init_results_folder():
    """初始化结果文件夹（仅创建，不清空）"""
    try:
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        print(f"[DEBUG] 结果图片实际保存路径: {RESULTS_FOLDER}", file=sys.stderr)  # 新增日志
        return True
    except Exception as e:
        print(f"文件夹初始化失败: {str(e)}", file=sys.stderr)
        sys.exit(1)


def save_results(original_img, segmented_img, gt_img=None):
    """保存结果到文件夹，返回包含路径的字典"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {}

    # 保存原图（标准化为uint8）
    original_norm = original_img
    if original_img.dtype != np.uint8:
        original_norm = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    original_path = os.path.join(RESULTS_FOLDER, f"{timestamp}_original.png")
    if not cv2.imwrite(original_path, original_norm):
        raise Exception(f"原图保存失败，路径：{original_path}")
    result["original"] = original_path

    # 保存分割结果
    segmented_path = os.path.join(RESULTS_FOLDER, f"{timestamp}_segmented.png")
    if not cv2.imwrite(segmented_path, segmented_img):
        raise Exception(f"分割结果保存失败，路径：{segmented_path}")
    result["segmented"] = segmented_path

    # 保存Ground Truth（如有）
    if gt_img is not None:
        gt_path = os.path.join(RESULTS_FOLDER, f"{timestamp}_ground_truth.png")
        if not cv2.imwrite(gt_path, gt_img):
            raise Exception(f"GT图保存失败，路径：{gt_path}")
        result["ground_truth"] = gt_path

    # 保存对比图
    fig, axes = plt.subplots(
        1,
        3 if gt_img is not None else 2,
        figsize=(18 if gt_img else 12, 6)
    )
    axes[0].imshow(segmented_img, cmap='gray')
    axes[0].set_title('分割结果')
    axes[0].axis('off')

    if gt_img is not None:
        axes[1].imshow(gt_img, cmap='gray')
        axes[1].set_title('真实标注')
        axes[1].axis('off')
        axes[2].imshow(original_norm, cmap='gray')
        axes[2].set_title('原始图像')
        axes[2].axis('off')
    else:
        axes[1].imshow(original_norm, cmap='gray')
        axes[1].set_title('原始图像')
        axes[1].axis('off')

    comparison_path = os.path.join(RESULTS_FOLDER, f"{timestamp}_comparison.png")
    try:
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        raise Exception(f"对比图保存失败：{str(e)}")
    finally:
        plt.close(fig)
    result["comparison"] = comparison_path

    return result


def load_model(device):
    """加载模型"""
    try:
        model = ResNetUNet(1, 1, pretrained=False)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        return model.to(device).eval()
    except Exception as e:
        print(f"模型加载失败: {str(e)}", file=sys.stderr)
        sys.exit(1)


def process_image(image_path, model, device):
    """处理图像（支持DICOM和普通格式）"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"输入图像不存在：{image_path}")

        # 读取图像
        if image_path.lower().endswith('.dcm'):
            dcm = pydicom.dcmread(image_path)
            img_array = dcm.pixel_array
            min_val = img_array.min()
            non_max_mask = img_array < img_array.max()
            if non_max_mask.any():
                max_val = img_array[non_max_mask].max()
            else:
                max_val = img_array.max()
            img_array = np.clip(img_array, min_val, max_val)
            img = Image.fromarray(img_array.astype(np.uint16)).convert('L')
        else:
            img = Image.open(image_path).convert('L')

        # 模型推理
        input_tensor = x_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = torch.sigmoid(model(input_tensor)).cpu().numpy()[0, 0]

        # 后处理
        seg_result = (output > 0.5).astype(np.uint8) * 255
        seg_result = cv2.resize(seg_result, (256, 256), cv2.INTER_NEAREST)

        return np.array(img), seg_result
    except Exception as e:
        print(f"图像处理失败: {str(e)}", file=sys.stderr)
        sys.exit(1)


def find_gt_image(image_path):
    """查找Ground Truth图像"""
    try:
        basename = os.path.basename(image_path)
        gt_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), "Ground", "Liver")
        gt_path = os.path.join(gt_dir, basename)
        if not os.path.exists(gt_path) and image_path.lower().endswith('.dcm'):
            gt_filename = os.path.splitext(basename)[0] + ".png"
            gt_path = os.path.join(gt_dir, gt_filename)
        return gt_path if os.path.exists(gt_path) else None
    except Exception as e:
        print(f"GT图像查找警告: {str(e)}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="肝脏图像分割")
    parser.add_argument("image_path", help="输入图像的完整路径（支持DICOM/PNG/JPG）")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"图像不存在: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    init_results_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", file=sys.stderr)
    model = load_model(device)
    original_img, seg_result = process_image(args.image_path, model, device)

    gt_img = None
    gt_path = find_gt_image(args.image_path)
    if gt_path:
        print(f"找到GT图像: {gt_path}", file=sys.stderr)
        gt_img = cv2.resize(np.array(Image.open(gt_path).convert('L')), (256, 256), cv2.INTER_NEAREST)
    else:
        print("未找到GT图像，仅生成原始图和分割图", file=sys.stderr)

    try:
        results = save_results(original_img, seg_result, gt_img)
        # 打印JSON结果，供后端解析
        print(json.dumps(results, ensure_ascii=False))
        print(f"结果保存完成，路径: {RESULTS_FOLDER}", file=sys.stderr)
    except Exception as e:
        print(f"结果保存失败: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # 将probs和targets展平
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        # 计算交集和并集
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, weight_bce=0.3, weight_dice=0.7):
        super(CombinedLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        # 调整logits尺寸以匹配targets
        if logits.shape[2:] != targets.shape[2:]:
            logits = F.interpolate(logits, size=targets.shape[2:], mode='bilinear', align_corners=False)
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.weight_bce * bce + self.weight_dice * dice


# ResNet编码器部分
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        
        # 修改第一层以适应单通道输入
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # 将预训练的RGB权重转换为单通道权重
            self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = x
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        return [c1, c2, c3, c4, c5]


class ResNetUNet(nn.Module):
    def __init__(self, in_ch, out_ch, pretrained=True):
        super(ResNetUNet, self).__init__()
        
        self.encoder = ResNetEncoder(pretrained=pretrained)
        
        # 中间连接层
        self.center = DoubleConv(512, 512)
        
        # 解码器部分 - 修正通道数以匹配ResNet特征
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # 编码器
        features = self.encoder(x)
        c1, c2, c3, c4, c5 = features
        
        # 中心连接
        center = self.center(c5)
        
        # 解码器
        up_6 = self.up6(center)
        # 确保up_6尺寸与c4匹配
        if up_6.size()[2:] != c4.size()[2:]:
            up_6 = F.interpolate(up_6, size=c4.size()[2:], mode='bilinear', align_corners=False)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        
        up_7 = self.up7(c6)
        # 确保up_7尺寸与c3匹配
        if up_7.size()[2:] != c3.size()[2:]:
            up_7 = F.interpolate(up_7, size=c3.size()[2:], mode='bilinear', align_corners=False)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        
        up_8 = self.up8(c7)
        # 确保up_8尺寸与c2匹配
        if up_8.size()[2:] != c2.size()[2:]:
            up_8 = F.interpolate(up_8, size=c2.size()[2:], mode='bilinear', align_corners=False)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        
        up_9 = self.up9(c8)
        # up_9尺寸与c1匹配
        if up_9.size()[2:] != c1.size()[2:]:
            up_9 = F.interpolate(up_9, size=c1.size()[2:], mode='bilinear', align_corners=False)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        
        c10 = self.conv10(c9)
        return c10

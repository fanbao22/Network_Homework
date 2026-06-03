import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import os



# 1. 手动实现核心量化函数
def linear_quantize(x, num_bits=8):
    """将浮点张量线性量化为整数张量 (Per-tensor 非对称量化)"""
    q_min = 0
    q_max = (1 << num_bits) - 1

    x_min = x.min().item()
    x_max = x.max().item()

    # 防止分母为0的情况（如全0张量）
    if x_min == x_max:
        return torch.zeros_like(x, dtype=torch.uint8), 1.0, 0

    # 计算 Scale 和 Zero Point
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = round(q_min - (x_min / scale))

    # 限制 Zero Point 在合法范围内
    zero_point = max(q_min, min(q_max, zero_point))

    # 线性映射并取整
    q_x = torch.round(x / scale) + zero_point

    # 截断超出范围的值，并转换为 UINT8
    q_x = torch.clamp(q_x, q_min, q_max).to(torch.uint8)

    return q_x, scale, zero_point


def linear_dequantize(q, scale, zero_point):
    """将整数张量反量化回浮点张量"""
    # 反量化公式： x' = (q - Z) * S
    x_dequant = (q.float() - zero_point) * scale
    return x_dequant



# 2. 数据处理和网络定义 (与作业2一致)
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

svhn_mean = (0.4377, 0.4438, 0.4728)
svhn_std = (0.1980, 0.2010, 0.1970)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(svhn_mean, svhn_std)
])


class ModifiedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)


def test_epoch(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = outputs.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    end_time = time.time()
    return 100 * correct / total, (end_time - start_time)


# 3. 测试
if __name__ == '__main__':
    # 1. 加载测试数据
    test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. 准备 FP32 基线模型
    print("加载 FP32 原始模型...")
    model_fp32 = ModifiedResNet18().to(DEVICE)
    # 读取作业2中训练得到的权重文件
    if os.path.exists("svhn_model.pth"):
        model_fp32.load_state_dict(torch.load("svhn_model.pth", map_location=DEVICE))
    else:
        print("警告: 未找到 svhn_model.pth。")

    fp32_acc, fp32_time = test_epoch(model_fp32, test_loader, DEVICE)
    print(f"FP32 基线模型 -> 测试准确率: {fp32_acc:.2f}%, 推理总耗时: {fp32_time:.4f}秒")

    # 3. 对模型权重进行手动量化 (Weight-only PTQ)
    print("\n开始进行 INT8 手动量化...")
    quantized_state_dict = {}
    quantization_params = {}

    for name, param in model_fp32.state_dict().items():
        # 只量化卷积层和全连接层的 weight，不量化 bias 和 BatchNorm 的统计量
        if 'weight' in name and param.dim() > 1:
            q_x, scale, zp = linear_quantize(param, num_bits=8)
            quantized_state_dict[name] = q_x
            quantization_params[name] = (scale, zp)
        else:
            quantized_state_dict[name] = param  # 其余参数原样保留

    # 保存量化后的字典以对比大小
    torch.save(quantized_state_dict, "svhn_model_int8.pth")

    # 4. 反量化并加载到模型中以测试精度 (伪量化/Fake Quantization)
    print("进行反量化以评估 INT8 精度损失...")
    fake_quant_state_dict = {}
    for name, param in model_fp32.state_dict().items():
        if name in quantization_params:
            q_x = quantized_state_dict[name]
            scale, zp = quantization_params[name]
            # 反量化回 FP32 以便让普通的 PyTorch 模型执行前向传播
            fake_quant_state_dict[name] = linear_dequantize(q_x, scale, zp)
        else:
            fake_quant_state_dict[name] = param

    model_fake_quant = ModifiedResNet18().to(DEVICE)
    model_fake_quant.load_state_dict(fake_quant_state_dict)

    int8_acc, int8_time = test_epoch(model_fake_quant, test_loader, DEVICE)
    print(f"INT8 量化模型 -> 测试准确率: {int8_acc:.2f}%, 推理总耗时: {int8_time:.4f}秒")

    # 5. 文件大小对比
    fp32_size = os.path.getsize("svhn_model.pth") / (1024 * 1024) if os.path.exists("svhn_model.pth") else 0
    int8_size = os.path.getsize("svhn_model_int8.pth") / (1024 * 1024)

    print("\n--- 评估总结 ---")
    print(f"FP32 模型大小: {fp32_size:.2f} MB")
    print(f"INT8 模型大小: {int8_size:.2f} MB (压缩率约 {fp32_size / int8_size:.2f}x)")
    print(f"准确率变化: {fp32_acc:.2f}% -> {int8_acc:.2f}% (差异: {int8_acc - fp32_acc:.2f}%)")
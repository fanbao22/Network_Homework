import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# 注意：导入专门为量化优化的 models 库
import torchvision.models.quantization as qmodels
import time
import os

BATCH_SIZE = 128
# 真实的 INT8 静态量化目前在 CPU 后端（x86或ARM）优化得最好
# GPU 不支持原生的 PyTorch Eager 模式静态量化
DEVICE = torch.device("cpu")

svhn_mean = (0.4377, 0.4438, 0.4728)
svhn_std = (0.1980, 0.2010, 0.1970)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(svhn_mean, svhn_std)
])


# 测试评估函数
def test_epoch(model, dataloader, device):
    # 对于真实的量化模型，不需要 to(device)，默认在 CPU 运行
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            pred = outputs.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    end_time = time.time()
    return 100 * correct / total, (end_time - start_time)


if __name__ == '__main__':
    # 1. 加载数据
    test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ==========================================
    # 2. 准备 FP32 基线模型 (使用可量化版本的 ResNet18)
    # ==========================================
    print("初始化量化版 ResNet18...")
    # quantize=False 表示先加载 FP32 版本
    model_fp32 = qmodels.resnet18(weights=None, quantize=False)

    # 同样为了适配 SVHN 的 32x32 小尺寸，修改网络头部
    model_fp32.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model_fp32.maxpool = nn.Identity()
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, 10)

    # ⚠️ 此时你可以加载你之前训练好的 FP32 权重
    if os.path.exists("svhn_model.pth"):
        model_fp32.load_state_dict(torch.load("svhn_model.pth", map_location='cpu'))

    model_fp32.eval()  # 量化前必须进入 eval 模式

    # 测一下基线速度
    fp32_acc, fp32_time = test_epoch(model_fp32, test_loader, DEVICE)
    print(f"【FP32 基线模型】 测试准确率: {fp32_acc:.2f}%, 推理总耗时: {fp32_time:.4f}秒")

    # ==========================================
    # 3. 真实的 PyTorch 静态量化流水线 (提速核心)
    # ==========================================
    print("\n开始执行真实的 INT8 静态量化加速...")

    # (A) 设置底层的量化计算引擎 (x86 电脑一般用 fbgemm 或 x86)
    # 如果你是 Mac M系列芯片 (ARM)，请将 'x86' 改为 'qnnpack'
    torch.backends.quantized.engine = 'x86'
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    # (B) 算子融合：将 Conv + BN + ReLU 融合成一个算子，极大减少内存访问
    model_fp32.fuse_model(is_qat=False)

    # (C) 准备模型：插入 Observer (观测器) 用于记录数据分布
    model_prepared = torch.ao.quantization.prepare(model_fp32, inplace=False)

    # (D) 模型校准 (Calibration)：跑少部分数据，让 Observer 计算出最佳的 Scale 和 Zero_point
    print("正在校准数据分布...")
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            model_prepared(x)
            if i >= 10:  # 跑 10 个 batch 足够校准了
                break

    # (E) 模型转换：正式将 FP32 算子替换为底层的 C++ INT8 真实算子！
    model_int8 = torch.ao.quantization.convert(model_prepared, inplace=False)

    # ==========================================
    # 4. 测试真实的 INT8 加速效果
    # ==========================================
    int8_acc, int8_time = test_epoch(model_int8, test_loader, DEVICE)
    print(f"【INT8 加速模型】 测试准确率: {int8_acc:.2f}%, 推理总耗时: {int8_time:.4f}秒")

    print("\n--- 终极加速报告 ---")
    print(f"提速比 (Speedup): {fp32_time / int8_time:.2f} 倍！")

    # 对比大小 (通过保存量化后的 TorchScript 引擎文件)
    torch.jit.save(torch.jit.script(model_int8), "svhn_model_int8_real.pth")
    fp32_size = os.path.getsize("svhn_model.pth") / (1024 * 1024) if os.path.exists("svhn_model.pth") else 0
    int8_size = os.path.getsize("svhn_model_int8_real.pth") / (1024 * 1024)
    print(f"文件压缩: {fp32_size:.2f} MB -> {int8_size:.2f} MB")
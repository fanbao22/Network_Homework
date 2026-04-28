import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. 数据预处理 (Data Preprocessing)
# ==========================================
# 要求: 确保使用的训练数据中都是固定格式的古诗 (如七言绝句)
def load_and_filter_data(json_pattern="poet.song.*.json"):
    poems = []
    # 读取所有匹配的 json 文件
    for file_path in glob.glob(json_pattern):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                paragraphs = item.get("paragraphs", [])
                # 过滤出七言绝句 (通常包含两句或四句，每句包含标点为8个字符，这里简化处理)
                # 确保格式整齐，去除包含异常字符的诗句
                poem_str = "".join(paragraphs)
                if len(poem_str) == 32 and "，" in poem_str and "。" in poem_str:
                    poems.append(poem_str)
    return poems


# 构建词汇表
def build_vocab(poems):
    vocab = set("".join(poems))
    vocab.add('<PAD>')  # 填充字符
    vocab.add('<START>')  # 起始字符
    vocab.add('<END>')  # 结束字符

    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return char2idx, idx2char, len(vocab)


class PoetryDataset(Dataset):
    def __init__(self, poems, char2idx):
        self.data = []
        for poem in poems:
            # 加上起始和结束标志
            idx_seq = [char2idx['<START>']] + [char2idx[c] for c in poem] + [char2idx['<END>']]
            self.data.append(idx_seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data[index]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


# ==========================================
# 2. 神经网络构建 (Neural Network Construction)
# ==========================================
# 要求: 使用 Pytorch 等神经网络框架构建 LSTM 等 RNN 类型网络
class PoetryLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(PoetryLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        out = self.fc(out)
        return out, hidden


# ==========================================
# 3. 训练与模型评估 (Training & Evaluation)
# ==========================================
def train_model():
    # 参数设置
    batch_size = 64
    embedding_dim = 128
    hidden_dim = 256
    num_epochs = 20
    learning_rate = 0.005
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 准备数据
    print("加载数据中...")
    # 注意：请确保你的 json 文件在同级目录，或者修改匹配路径
    poems = load_and_filter_data("*.json")

    # 如果没找到数据，生成一些假数据以供测试代码逻辑
    if not poems:
        print("未检测到JSON文件，使用测试数据...")
        poems = ["太道既如砥，安寧險艱。通溝水決決，出林鳥關關。"] * 100

    char2idx, idx2char, vocab_size = build_vocab(poems)
    dataset = PoetryDataset(poems, char2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = PoetryLSTM(vocab_size, embedding_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []

    print("开始训练...")
    for epoch in range(num_epochs):
        total_loss = 0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output, _ = model(x)

            # reshape 为 (batch_size * seq_len, vocab_size) 方便计算 loss
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f} ====")

    # 要求: 绘制一张 loss 图展示构建的神经网络可收敛
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', color='blue', label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, num_epochs + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('training_loss_curve.png')
    plt.show()

    return model, char2idx, idx2char, device


# ==========================================
# 4. 诗句生成 (Poetry Generation)
# ==========================================
# 要求: 以“明月”为总起词，生成固定格式的古诗
def generate_poetry(model, start_words, char2idx, idx2char, device, max_len=32):
    model.eval()
    hidden = None

    # 将输入词转化为 tensor
    input_seq = [char2idx.get('<START>')] + [char2idx.get(c, char2idx.get('<PAD>')) for c in start_words]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    generated_poem = start_words

    with torch.no_grad():
        for _ in range(max_len - len(start_words)):
            output, hidden = model(input_tensor, hidden)
            # 取最后一个时间步的输出
            last_out = output[:, -1, :]
            # 使用 argmax 获取最可能的下一个词
            predicted_idx = torch.argmax(last_out, dim=-1).item()
            predicted_char = idx2char[predicted_idx]

            if predicted_char == '<END>':
                break

            generated_poem += predicted_char

            # 将预测出的词作为下一步的输入
            input_tensor = torch.tensor([[predicted_idx]], dtype=torch.long).to(device)

    return generated_poem


if __name__ == '__main__':
    # 1. 训练模型并绘制 Loss 图
    model, char2idx, idx2char, device = train_model()

    # 2. 生成古诗 (以“明月”为总起词)
    print("\n【生成演示】:")
    start_phrase = "明月"
    poem = generate_poetry(model, start_phrase, char2idx, idx2char, device)
    print(poem)
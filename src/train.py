'''
    模型训练脚本
'''

import time
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config import *
from model import TranslationModel
from dataset import get_dataloader
from tokenizer import ZhTokenizer, EnTokenizer

# 整体训练流程
def train():
    print("模型开始训练...")
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 获取训练集加载器
    train_loader = get_dataloader(train=True)
    # 3. 定义模型
    # 3.1 先创建分词器，为了获取词表大小
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR / ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR / EN_VOCAB_FILE)
    # 3.2 定义模型
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_id,
        en_tokenizer.pad_id,
    ).to(device)
    # 4. 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 5. 定义训练日志写入器
    writer = SummaryWriter(LOGS_DIR / time.strftime("%Y_%m_%d-%H_%M_%S"))
    # 6. 开始训练
    min_loss = float("inf")
    for epoch in range(EPOCHS):
        # 训练一个epoch，返回平均损失
        print(f"=============Epoch: {epoch + 1}=============")
        this_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Loss: {this_loss:.4f}")
        # 将当前损失写入训练日志
        writer.add_scalar("loss", this_loss, epoch + 1)
        # 保存模型
        if this_loss < min_loss:
            min_loss = this_loss
            torch.save(model.state_dict(), MODELS_DIR/BEST_MODEL)
            print("模型保存成功！")

    writer.close()
    print("模型训练成功！")

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0  # 总损失
    for inputs, targets in tqdm(train_loader, desc='训练'):
        # 加载到设备，形状 inputs -> (N, S), targets -> (N, T+1)
        inputs, targets = inputs.to(device), targets.to(device)
        # 0. 准备需要传入的参数
        # 0.1 截取解码器的输入(N, T)以及最终的目标序列(N, T)
        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:]
        # 0.2 定义掩码
        tgt_mask = model.transformer.generate_square_subsequent_mask(
            decoder_inputs.shape[1]
            # decoder_inputs.shape[0]  # N，batch大小
            # decoder_inputs.shape[1]  # T，序列长度
        ).bool().to(device)
        src_pad_mask = (inputs == model.src_embedding.padding_idx)
        tgt_pad_mask = (decoder_inputs == model.tgt_embedding.padding_idx)

        # 1. 前向传播，得到预测输出，形状 (N, T, tgt_vocab_size)
        decoder_outputs = model(
            inputs,
            decoder_inputs,
            tgt_mask=tgt_mask,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
        )

        # 2. 计算损失，注意需要将预测输出转置，形状(N, C=tgt_vocab_size, T)
        loss = loss_fn(decoder_outputs.mT, decoder_targets)

        # 3. 反向传播，计算梯度
        loss.backward()

        # 4. 更新参数
        optimizer.step()

        # 5. 梯度清零
        optimizer.zero_grad()

        # 6. 累加总损失
        total_loss += loss.item()

    # 返回总损失
    return total_loss / len(train_loader)

if __name__ == "__main__":
    train()
'''
    定义模型结构：基于Transformer的LM
'''

import math
import torch
import torch.nn as nn
from config import *

# 定义位置编码层类
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=MAX_SEQ_LEN, d_model=D_MODEL):
        super().__init__()
        # 定义位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 预计算：遍历每个位置(行)，进行编码计算
        for pos in range(max_len):
            for _2i in range(0, d_model, 2):
                # 代入公式计算
                pe[pos, _2i] = math.sin(pos / (10000 ** (2 * _2i / d_model)))
                pe[pos, _2i + 1] = math.cos(pos / (10000 ** (2 * _2i / d_model)))
        self.register_buffer('pe', pe)

    # 向前传播，传入词向量列表(N, L, d_model)
    def forward(self, x):
        # 截取pe矩阵的前L行，形状(L, d_model)
        part_pe = self.pe[: x.shape[-2]]
        # 广播之后与词向量叠加，返回
        return x + part_pe
        # x:       (N, L, d_model)
        # part_pe:    (L, d_model)   ← 自动在最前面补一个维度变成 (1, L, d_model)
        # 结果:    (N, L, d_model)

# 定义模型类
class TranslationModel(nn.Module):
    # 初始化
    def __init__(self, src_vocab_size, tgt_vocab_size, src_padding_idx=0, tgt_padding_idx=0):
        super().__init__()
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, D_MODEL, padding_idx=src_padding_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, D_MODEL, padding_idx=tgt_padding_idx)
        # 位置编码层
        self.positional_encoding = PositionalEncoding(MAX_SEQ_LEN, D_MODEL)
        # Transformer层
        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            batch_first=True
        )
        # 输出线性层
        '''
        解码器输出:  (N, T, D_MODEL)
                nn.Linear(D_MODEL, tgt_vocab_size)
        输出:       (N, T, tgt_vocab_size)
        '''
        self.out = nn.Linear(D_MODEL, tgt_vocab_size)

    # 前向传播
    def forward(self, input_ids, decoder_input_ids, tgt_mask, src_pad_mask, tgt_pad_mask):
        # 编码
        memory = self.encode(input_ids, src_pad_mask)
        # 解码
        output = self.decode(decoder_input_ids, memory, tgt_mask, tgt_pad_mask, src_pad_mask)
        return output

    # 编码，传入 input_ids，形状(N,S)
    def encode(self, input_ids, src_pad_mask):
        # 1. 词表示，得到词向量，形状(N, S, d_model)
        embedding = self.src_embedding(input_ids)
        # 2. 位置编码，形状(N, S, d_model)
        src = self.positional_encoding(embedding)
        # 3. Transformer编码器前向传播，形状(N, S, d_model)
        memory = self.transformer.encoder(src, src_key_padding_mask=src_pad_mask)
        return memory

    # 解码，传入decoder_input_ids，形状(N, T)；以及memory
    def decode(self, decoder_input_ids, memory, tgt_mask, tgt_pad_mask, memory_pad_mask):
        # 1. 词表示，得到词向量，形状(N, T, d_model)
        embedding = self.tgt_embedding(decoder_input_ids)
        # 2. 位置编码，形状(N, T, d_model)
        tgt = self.positional_encoding(embedding)
        # 3. Transformer解码器前向传播，形状(N, T, d_model)
        output = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,  # 因果掩码，防止偷看未来
            # 同样在解码器自注意力里，让其他位置不要去关注目标序列末尾那些<pad>占位符。
            # 它和tgt_mask一起作用在同一个注意力上，被屏蔽的是两者的并集。
            tgt_key_padding_mask=tgt_pad_mask,
            # 源序列的pad掩码
            memory_key_padding_mask=memory_pad_mask
        )
        # 4. 输出线性层整合特征，得到词表分布向量，形状(N, T, tgt_vocab_size)
        output = self.out(output)
        return output

if __name__ == '__main__':

    # 定义基本参数
    N = 64  # batch大小，一批多少句
    S = 20  # 源序列长度，即中文每句多少个token
    T = 15  # 目标序列长度(解码器输入，英文)
    src_vocab_size = 1200
    tgt_vocab_size = 1500

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义输入数据
    input_ids = torch.randint(src_vocab_size, (N, S)).to(device)
    decoder_input_ids = torch.randint(tgt_vocab_size, (N, T)).to(device)

    # 定义模型
    model = TranslationModel(src_vocab_size, tgt_vocab_size).to(device)

    # 定义掩码
    # 生成解码器自注意力用的因果掩码（防止偷看未来的词）
    tgt_mask = model.transformer.generate_square_subsequent_mask(T).bool().to(device)
    src_padding_idx = 0
    tgt_padding_idx = 0
    src_pad_mask = (input_ids == src_padding_idx)
    tgt_pad_mask = (decoder_input_ids == tgt_padding_idx)

    # 前向传播
    output = model(input_ids, decoder_input_ids, tgt_mask, src_pad_mask, tgt_pad_mask)
    print(output.shape)
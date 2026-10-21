# 中译英神经机器翻译（Transformer）

简体中文 | [English](README.md)

基于 PyTorch `nn.Transformer` 实现的轻量级中文 → 英文机器翻译项目，涵盖完整流程：数据预处理、词表构建、模型训练、BLEU 评估，以及交互式命令行翻译程序。

## 特性

- 带正弦位置编码的 Encoder–Decoder Transformer
- 中文按字符分词，英文使用 NLTK Treebank 分词器
- 使用 Teacher Forcing 训练，配合 padding 掩码与因果掩码
- 自回归贪心解码，支持批量推理
- 基于 NLTK 的语料级 BLEU 评估
- 使用 TensorBoard 记录训练损失
- 交互式命令行翻译小程序

## 项目结构

`config.py` 中的 `ROOT_DIR` 取的是脚本所在目录的**上一级**，因此脚本需要放在子目录（如 `src/`）中：

```
project_root/
├── src/                  # 所有 .py 文件
│   ├── config.py         # 路径、特殊 token、超参数
│   ├── tokenizer.py      # 中/英文分词器，词表构建与加载
│   ├── preprocess.py     # 读取原始数据、划分、编码并保存为 .jsonl
│   ├── dataset.py        # 数据集、对齐函数（填充）、数据加载器
│   ├── model.py          # 位置编码 + TranslationModel
│   ├── train.py          # 训练流程，按训练损失保存最佳模型
│   ├── evaluate.py       # 在测试集上计算 BLEU
│   └── predict.py        # 批量推理 + 交互式命令行程序
├── data/
│   ├── raw/              # cmn.txt（制表符分隔：英文 \t 中文 [\t ...]）
│   └── processed/        # train.jsonl、test.jsonl（自动生成）
├── models/               # best_model.pt、zh_vocab.txt、en_vocab.txt（自动生成）
└── logs/                 # TensorBoard 日志（自动生成）
```

> 请提前创建 `data/raw`、`data/processed`、`models`、`logs` 目录——脚本不会自动创建目录。

## 环境依赖

- Python 3.12+
- PyTorch（有 GPU 时自动使用）
- pandas、scikit-learn、nltk、tqdm、tensorboard

```bash
pip install -r requirements.txt
```

## 数据

将 `cmn.txt`（Tatoeba 风格的中英句对，制表符分隔）放入 `data/raw/`。程序只读取前两列（`en`、`zh`），并按 80% / 20% 划分训练集和测试集（`random_state=42`）。

## 使用方法

以下命令均在 `src/` 目录下运行。

```bash
# 1. 数据预处理：基于训练集构建词表，并将文本编码为 id
python preprocess.py

# 2.（可选）检查数据加载器 / 模型张量形状
python dataset.py
python model.py

# 3. 训练
python train.py

# 4. 评估（测试集 BLEU）
python evaluate.py

# 5. 交互式翻译
python predict.py
```

使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir ../logs
```

在交互程序中输入中文句子后回车即可翻译；输入 `q`、`quit` 或 `exit` 退出。

## 配置项

所有配置都在 `config.py` 中：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `BATCH_SIZE` | 64 | 批大小 |
| `EPOCHS` | 50 | 训练轮数 |
| `LEARNING_RATE` | 0.001 | Adam 学习率 |
| `MAX_SEQ_LEN` | 128 | 位置编码最大长度 / 最大生成 token 数 |
| `D_MODEL` | 128 | 词向量 / 隐藏层维度 |
| `NUM_HEADS` | 4 | 注意力头数 |
| `NUM_ENCODER_LAYERS` | 2 | 编码器层数 |
| `NUM_DECODER_LAYERS` | 2 | 解码器层数 |

## 工作原理

1. **分词**：中文按单个字符切分；英文使用 `TreebankWordTokenizer`（解码时使用 `TreebankWordDetokenizer`）。词表以 `<pad>`、`<unk>`、`<sos>`、`<eos>` 开头；英文目标序列会被包装为 `<sos> ... <eos>`。
2. **训练**：对长度为 T+1 的目标序列，解码器输入为 `targets[:, :-1]`，标签为 `targets[:, 1:]`；交叉熵损失忽略 `<pad>`。平均损失最低的 epoch 模型会保存到 `models/best_model.pt`。
3. **推理**：编码器只运行一次；解码器从 `<sos>` 开始逐 token 贪心生成，直到所有句子都输出 `<eos>` 或达到 `MAX_SEQ_LEN`。
4. **评估**：参考译文去掉 `<sos>`/`<eos>` 后，对 token id 序列计算 `corpus_bleu`。

## 注意事项与局限

- 模型选择依据的是**训练**损失，没有独立的验证集，训练 50 个 epoch 可能出现过拟合。
- 解码方式为贪心解码（未使用 beam search）。
- 中文为字符级、英文为词级词表（无子词切分），低频词会变成 `<unk>`。
- `evaluate.py` / `predict.py` 中的 `torch.load` 使用默认设备加载权重；若在 GPU 上训练、在 CPU 上推理，请传入 `map_location`。
- BLEU 使用 NLTK 默认的 4-gram 权重且未做平滑，短句上得分可能偏低甚至为 0。

## 许可证

MIT

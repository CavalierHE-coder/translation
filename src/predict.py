'''
    推理预测脚本及中英翻译应用程序
'''
from gc import is_finalized

import torch
from config import *
from model import TranslationModel
from tokenizer import ZhTokenizer,EnTokenizer

# 推理预测主流程（接口）：传入用户输入文本（中文句子），返回对应的英文翻译
def predict(text, model, src_tokenizer, tgt_tokenizer, device) -> str:
    # 1. 文本表示（中文），得到模型输入
    # 1.1 编码，得到中文id列表
    ids = src_tokenizer.encode(text)
    # 1.2 转换成Tensor，作为模型输入(N=1, L)
    inputs = torch.tensor([ids])

    # 2. 推理预测，得到目标id列表（英文），形如 [[3, 12, 51, 8]]
    # ( 如 N = 2 则为 [[3, 12, 51, 8],[42, 79, 13, 90, 123]] )
    results = predict_batch(model, inputs, tgt_tokenizer, device)

    # 3. 解码，转换成目标文本（英文句子）
    sentence = tgt_tokenizer.decode(results[0])
    return sentence

# 核心逻辑
def predict_batch(model, inputs, tgt_tokenizer, device):
    model.eval()
    with torch.no_grad():
        # 输入数据形状(N, S)
        inputs = inputs.to(device)
        # 前向传播
        # 1. 编码，得到(N, S, d_model)
        src_pad_mask = (inputs == model.src_embedding.padding_idx)
        memory = model.encode(inputs, src_pad_mask)

        # 2. 解码（自回归生成）
        # 2.1 构建初始解码器输入，形状(N, T=1)，N个<sos>
        N = inputs.shape[0]
        decoder_inputs = torch.full((N, 1), tgt_tokenizer.sos_id).to(device)

        # 记录每条数据是否生成结束，初始值N个False
        is_finished = torch.full((N,), False).to(device)

        # 2.2 循环进行前向传播（自回归生成）
        generated_ids = []  # 保存生成id的列表
        for i in range(MAX_SEQ_LEN):
            # 2.2.1 前向传播，得到输出 (N, T, tgt_vocab_size)
            tgt_mask = model.transformer.generate_square_subsequent_mask(
                decoder_inputs.shape[1]
            ).bool().to(device)
            tgt_pad_mask = (decoder_inputs == model.tgt_embedding.padding_idx)
            decoder_outputs = model.decode(
                decoder_inputs,
                memory,
                tgt_mask,
                tgt_pad_mask,
                src_pad_mask,
            )

            # 2.2.2 提取最后一个位置的向量 (N, tgt_vocab_size)，贪心解码得到预测结果id，形状 (N, 1)
            next_token_ids = torch.argmax(decoder_outputs[:, -1], dim=-1).unsqueeze(-1)

            # 2.2.3 更新解码器输入，序列长度加1（T->T+1），拼接后的形状 (N, T+1)
            decoder_inputs = torch.cat([decoder_inputs, next_token_ids], dim=-1)

            # 2.2.4 保存结果到列表中
            generated_ids.append(next_token_ids)

            # 2.2.5 如果出现<eos>，就退出
            is_finished |= (next_token_ids.squeeze(-1) == tgt_tokenizer.eos_id)
            if is_finished.all():
                break

        # 2.3 处理生成结果，得到每一句对应的英文翻译（id列表的列表）
        # 2.3.1 合并生成结果 (N, L)，并转为二维列表
        generated_ids_list = torch.cat(generated_ids, dim=-1).tolist()

        # 2.3.2 截取有效生成部分：找到<eos>，去掉之后的所有token_id
        for index, ids in enumerate(generated_ids_list):
            # 判断是否存在<eos>
            if tgt_tokenizer.eos_id in ids:
                # 找到<eos>的位置（索引号）
                pos_eos = ids.index(tgt_tokenizer.eos_id)
                # 截断
                generated_ids_list[index] = ids[:pos_eos]

        return generated_ids_list

# 构建应用程序
def run_app():
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 创建分词器
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR / ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR / EN_VOCAB_FILE)

    # 3. 定义并加载训练好的模型
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_id,
        en_tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(torch.load(MODELS_DIR/BEST_MODEL))
    print("模型加载成功")

    # 应用程序
    print("欢迎使用中英翻译小程序！输入 q/quit/exit 退出...")
    # 核心是一个死循环
    while True:
        # 等待用户输入中文文本
        user_input = input("中文：")
        # 处理特殊输入
        if user_input in ["q", "quit", "exit"]:
            print("欢迎下次使用！")
            break

        if user_input.strip() == '':
            print("请输入有效中文内容...")
            continue

        sentence = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print("英文：", sentence)

if __name__ == '__main__':
    run_app()
'''
    模型评估脚本
'''

import torch
from nltk.translate.bleu_score import corpus_bleu  # 计算BLEU指标
from tqdm import tqdm
from config import *
from model import TranslationModel
from dataset import get_dataloader
from tokenizer import ZhTokenizer, EnTokenizer
from predict import predict_batch


def evaluate(model, test_loader, tgt_tokenizer, device):
    model.eval()

    # 用列表保存所有数据的预测结果和正确结果
    list_of_references = []
    hypothesis = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='评估'):
            inputs, targets = inputs.to(device), targets.tolist()
            # 1. 推理预测，得到预测结果
            batch_result = predict_batch(model,inputs,tgt_tokenizer, device)
            # 2. 合并到列表中
            hypothesis.extend(batch_result)
            # 3. 将参考译文结果截去<sos>和<eos>后，添加到列表
            list_of_references.extend([ [target[1:target.index(tgt_tokenizer.eos_id)]] for target in targets ])

    return corpus_bleu(list_of_references, hypothesis)

# 评估整体流程
def run_evaluate():
    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 创建分词器
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR / ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR / EN_VOCAB_FILE)

    # 3. 定义并加载训练号的模型
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_id,
        en_tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / BEST_MODEL))
    print("模型加载成功！")

    # 4. 获取测试集加载器
    test_loader = get_dataloader(train=False)

    # 5. 评估
    bleu = evaluate(model, test_loader, en_tokenizer, device)

    print("BLEU:", bleu)

if __name__ == '__main__':
    run_evaluate()
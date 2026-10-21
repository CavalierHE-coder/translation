'''
    数据预处理脚本
'''
import pandas as pd
from sklearn.model_selection import train_test_split    # 划分训练集和测试集

from config import *
from tokenizer import ZhTokenizer, EnTokenizer  # 中英文分词器

def preprocess():
    print("---------开始数据预处理---------")

    # 1. 读取原始数据文件
    df = pd.read_csv(
        RAW_DATA_DIR/RAW_DATA_FILE,
        sep='\t',    # 分隔符
        header=None,    # 没有列名
        usecols=[0, 1], # 只选取前两列
        names=['en', 'zh'],     # 指定列名
        encoding='utf-8'
    )
    print(df.head())

    # 2. 数据清洗
    df.dropna(inplace=True)

    # 3. 划分数据集（训练集和测试集）
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(len(train_df), len(test_df))

    # 4. 文本表示（分词器操作）
    # 4.0 基于训练语料，生成词表
    ZhTokenizer.build_vocab(train_df['zh'].tolist(), MODELS_DIR/ZH_VOCAB_FILE)
    EnTokenizer.build_vocab(train_df['en'].tolist(), MODELS_DIR/EN_VOCAB_FILE)

    # 4.1 创建分词器
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR / ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR / EN_VOCAB_FILE)

    # 4.2 编码，得到id列表
    zh_encode = lambda text: zh_tokenizer.encode(text)
    en_encode = lambda text: en_tokenizer.encode(text, add_special_tokens=True)
    train_df['zh'] = train_df['zh'].apply(zh_encode)
    train_df['en'] = train_df['en'].apply(en_encode)
    test_df['zh'] = test_df['zh'].apply(zh_encode)
    test_df['en'] = test_df['en'].apply(en_encode)

    # 5. 保存数据集到文件
    train_df.to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True)
    test_df.to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print("---------数据预处理完成---------")

if __name__ == '__main__':
    preprocess()
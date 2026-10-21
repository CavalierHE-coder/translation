'''
    配置文件
'''
from pathlib import Path

# 目录路径
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
LOGS_DIR = ROOT_DIR / 'logs'

# 文件名
RAW_DATA_FILE = 'cmn.txt'   # 原始数据文件
TRAIN_DATA_FILE = 'train.jsonl'   # 训练集数据文件
TEST_DATA_FILE = 'test.jsonl'     # 测试集数据文件

BEST_MODEL = 'best_model.pt'    # 保存最佳模型参数
ZH_VOCAB_FILE = 'zh_vocab.txt'      # 中文词表文件
EN_VOCAB_FILE = 'en_vocab.txt'      # 英文词表文件

# 特殊token
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

# 训练超参数
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 128  # 限制模型能处理/生成的序列最多有多少个token

# 模型结构超参数
D_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
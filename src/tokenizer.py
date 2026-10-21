'''
    自定义分词器
'''
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
from config import *

# 基类：通用分词器
class BaseTokenizer:
    # 初始化，传入词的列表
    def __init__(self, vocab_list):
        self.vocab_size = len(vocab_list)
        self.id2word = vocab_list
        self.word2id = { word:id for id,word in enumerate(self.id2word) }
        # 特殊token的id
        self.unk_id = self.word2id[UNK_TOKEN]
        self.pad_id = self.word2id[PAD_TOKEN]
        self.sos_id = self.word2id[SOS_TOKEN]
        self.eos_id = self.word2id[EOS_TOKEN]

    # 创建词表：输入语料数据（句子的列表），以及保存词表文件的路径
    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        # 定义词的集合，用于去重
        vocab_set = set()
        # 遍历每个句子，分词，并添加到集合中
        for sentence in sentences:
            tokens = cls.tokenize(sentence)
            vocab_set.update(tokens)
        # 添加特殊token，转换成列表
        vocab_list = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(vocab_set)
        # 保存到文件
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    # 创建实例对象方法：从词表文件中加载词的列表，初始化对象并返回
    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [ token.strip() for token in f.readlines()]
        return cls(vocab_list)

    # 分词，文本 → 词的列表
    @classmethod
    def tokenize(cls, text) -> list[str]:
        pass

    # 编码，文本 → id的列表
    def encode(self, text, add_special_tokens=False):
        # 1. 分词
        tokens = self.tokenize(text)
        # 2. 查词表得到id
        ids = [ self.word2id.get(token, self.unk_id) for token in tokens ]
        # 3. 判断，是否需要添加特殊标记<sos>和<eos>
        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]

        return ids

    # 解码，id的列表 → 文本（目标句子）
    def decode(self, ids) -> str:
        pass

# 中文分词器
class ZhTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return list(text)

# 英文分词器
class EnTokenizer(BaseTokenizer):
    # 定义类属性
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls.tokenizer.tokenize(text)

    # 解码，id的列表 → 文本（目标句子）
    def decode(self, ids) -> str:
        # 1. 查词表，将id转换成token
        tokens = [ self.id2word[id] for id in ids ]
        # 2. 合成句子
        sentence = self.detokenizer.detokenize(tokens)
        return sentence

if __name__ == '__main__':
    zh_text = "我爱自然语言处理！"
    en_text = "I love NLP!"

    # 创建分词器
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR/ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR/EN_VOCAB_FILE)

    print("中文词表大小：", zh_tokenizer.vocab_size)
    print("英文词表大小：", en_tokenizer.vocab_size)

    # 编码
    zh_ids = zh_tokenizer.encode(zh_text)
    en_ids = en_tokenizer.encode(en_text, add_special_tokens=True)

    print("中文id列表：", zh_ids)
    print("英文id列表：", en_ids)
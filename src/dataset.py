"""
    创建数据集和数据加载器
"""

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from config import *

# 自定义数据集类
class TranslationDataset(Dataset):
    # 初始化
    def __init__(self, data_path):
        # 读取数据文件，转换为字典的列表
        self.data = pd.read_json(data_path, lines=True).to_dict(orient='records')

    # 获取数据集大小
    def __len__(self):
        return len(self.data)

    # 根据索引号，获取数据元素：二元组 (input, target)
    def __getitem__(self, idx):
        item = self.data[idx]  # 一个数据元素，字典：{'en':[], 'zh': []}
        input = torch.tensor(item['zh'])
        target = torch.tensor(item['en'])
        return input, target

# 定义对齐函数（批量数据填充），传入batch：二元组的列表 [(input1,targets), (input2, target2), ...]
def collate_fn(batch):
    # 拆分input和target列表
    input_list = [ item[0] for item in batch ]
    target_list = [ item[1] for item in batch ]
    # 分别填充，合并成一批数据的Tensor
    input_batch = pad_sequence(input_list, batch_first=True)
    target_batch = pad_sequence(target_list, batch_first=True)
    return input_batch, target_batch

# 定义获取加载器的方法，传入参数确定训练集/测试集
def get_dataloader(train=True):
    # 获取文件路径
    data_path = PROCESSED_DATA_DIR / (TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    # 创建数据集
    dataset = TranslationDataset(data_path)
    # 定义数据加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    # 获取数据集
    # train_dataset = TranslationDataset(PROCESSED_DATA_DIR/TRAIN_DATA_FILE)
    # print("训练集大小：", len(train_dataset))
    # print("第一条训练数据：", train_dataset[0])
    # print("第二条训练数据：", train_dataset[1])

    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)

    print("训练集第一批数据形状：")
    for input, target in train_loader:
        # input shape: torch.Size([64, 19]) 这批64条中文里最长的有19个token
        print("input shape: ", input.shape)
        print("target shape: ", target.shape)
        break
    print("测试集第一批数据形状：")
    for input, target in test_loader:
        print("input shape: ", input.shape)
        print("target shape: ", target.shape)
        break
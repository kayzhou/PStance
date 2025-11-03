import torch
from torch.utils.data import TensorDataset, DataLoader
# 使用modelscope来加载分词器
import modelscope
from modelscope import AutoTokenizer


"""
将目标和文本转换为模型所需的ID格式

参数:
    tokenizer: 用于文本编码的分词器
    target: 目标实体列表
    text: 文本列表

返回:
    tuple: 包含input_ids, seg_ids, attention_masks, sent_len的元组
"""
def convert_data_to_ids(tokenizer, target, text):
    
    # 初始化结果列表
    input_ids = []      # 存储token IDs
    seg_ids = []        # 存储分段IDs，区分目标和文本
    attention_masks = [] # 存储注意力掩码
    sent_len = []       # 存储有效序列长度
    
    # 遍历每个目标-文本对
    for tar, sent in zip(target, text):
        # 使用分词器编码目标和文本
        encoded_dict = tokenizer.encode_plus(
                            ' '.join(tar),                  # 目标实体
                            ' '.join(sent),                 # 输入文本
                            add_special_tokens=True,        # 添加[CLS]和[SEP]特殊标记
                            max_length=128,                 # 最大序列长度
                            padding='max_length',           # 填充到最大长度
                            return_attention_mask=True,     # 返回注意力掩码
                       )

        # 将编码后的结果添加到列表中
        input_ids.append(encoded_dict['input_ids'])
        seg_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        sent_len.append(sum(encoded_dict['attention_mask']))  # 计算有效序列长度
    
    return input_ids, seg_ids, attention_masks, sent_len
    

"""
处理BERT/BERTweet模型的输入数据

参数:
    x_train_all: 训练集数据 [文本, 标签, 目标]
    x_val_all: 验证集数据 [文本, 标签, 目标]
    x_test_all: 测试集数据 [文本, 标签, 目标]
    model_select: 模型选择，'Bertweet'或'Bert'

返回:
    tuple: 处理后的训练集、验证集、测试集数据
"""
def data_helper_bert(x_train_all, x_val_all, x_test_all, model_select):
    
    print('Loading data')
    
    # 解包数据
    x_train, y_train, x_train_target = x_train_all[0], x_train_all[1], x_train_all[2]
    x_val, y_val, x_val_target = x_val_all[0], x_val_all[1], x_val_all[2]
    x_test, y_test, x_test_target = x_test_all[0], x_test_all[1], x_test_all[2]
                                                          
    # 打印数据集基本信息
    print(f"Length of x_train: {len(x_train)}, the sum is: {sum(y_train)}")
    print(f"Length of x_val: {len(x_val)}, the sum is: {sum(y_val)}")
    print(f"Length of x_test: {len(x_test)}, the sum is: {sum(y_test)}")
    
    # 根据模型选择加载对应的分词器（使用modelscope）
    if model_select == 'Bertweet':
        # 使用本地缓存的模型kayzhou/bertweet-base
        tokenizer = AutoTokenizer.from_pretrained("kayzhou/bertweet-base", normalization=True)
    elif model_select == 'Bert':
        # 使用modelscope的bert-base模型
        tokenizer = AutoTokenizer.from_pretrained("kayzhou/bert-base", do_lower_case=True)
        
    # 对训练集、验证集和测试集进行编码
    x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len = \
                    convert_data_to_ids(tokenizer, x_train_target, x_train)
    x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len = \
                    convert_data_to_ids(tokenizer, x_val_target, x_val)
    x_test_input_ids, x_test_seg_ids, x_test_atten_masks, x_test_len = \
                    convert_data_to_ids(tokenizer, x_test_target, x_test)

    # 重新组织数据
    x_train_all = [x_train_input_ids, x_train_seg_ids, x_train_atten_masks, y_train, x_train_len]
    x_val_all = [x_val_input_ids, x_val_seg_ids, x_val_atten_masks, y_val, x_val_len]
    x_test_all = [x_test_input_ids, x_test_seg_ids, x_test_atten_masks, y_test, x_test_len]
    
    return x_train_all, x_val_all, x_test_all


"""
创建数据加载器

参数:
    x_all: 处理后的数据 [input_ids, seg_ids, attention_masks, labels, lengths]
    batch_size: 批次大小
    data_type: 数据类型，'train'或其他（验证/测试）

返回:
    tuple: 包含tensor和数据加载器的元组
"""
def data_loader(x_all, batch_size, data_type):
    
    # 自动选择设备（CPU或GPU），确保代码在CPU上也能执行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将数据转换为PyTorch张量并移至指定设备
    x_input_ids = torch.tensor(x_all[0], dtype=torch.long).to(device)
    x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).to(device)
    x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).to(device)
    y = torch.tensor(x_all[3], dtype=torch.long).to(device)
    x_len = torch.tensor(x_all[4], dtype=torch.long).to(device)

    # 创建数据集
    tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, y, x_len)
    
    # 创建数据加载器，训练集打乱，验证/测试集不打乱
    if data_type == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

    return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader
    

"""
将统一测试集分割为三个子测试集（针对特朗普、拜登和伯尼）

参数:
    input_data: 合并的测试数据

返回:
    list: 包含三个子测试集的列表
"""
def sep_test_set(input_data):
    
    # 分割组合测试集为特朗普、拜登和伯尼三个部分
    # 根据数据集的特定格式，按照固定索引进行分割
    data_list = [
        input_data[:777],        # 特朗普测试集
        input_data[777:1522],    # 拜登测试集
        input_data[1522:2157]    # 伯尼测试集
    ]
    
    return data_list

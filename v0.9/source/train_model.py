import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
import json
import utils.preprocessing as pp  # 导入数据预处理模块
import utils.data_helper as dh  # 导入数据辅助函数
# from transformers.optimization import AdamW
from torch.optim import AdamW  # 新版推荐方式导入优化器
from utils import modeling, model_eval  # 导入模型定义和评估模块

# 测试
"""
运行立场分类器模型的主函数
处理数据加载、预处理、模型训练和评估的完整流程
"""
def run_classifier():
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_target", type=str, default="trump", help="目标实体")
    parser.add_argument("--model_select", type=str, default="Bertweet", help="BERTweet or BERT model")
    parser.add_argument("--train_mode", type=str, default="adhoc", help="unified or adhoc")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    args = parser.parse_args()

    # 设置随机种子列表，用于重复实验
    random_seeds = [0, 1, 14, 15, 16, 17, 19]
    target_word_pair = [args.input_target]  # 目标实体列表
    model_select = args.model_select  # 选择的模型
    train_mode = args.train_mode  # 训练模式
    lr = args.lr  # 学习率
    batch_size = args.batch_size  # 批次大小
    total_epoch = args.epochs  # 总训练轮数
    
    # 创建规范化词典，用于文本预处理中的词汇替换
    with open("../data/noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("../data/emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    # 合并两个词典
    normalization_dict = {**data1, **data2}

    # 遍历所有目标实体
    for target_index in range(len(target_word_pair)):
        best_result, best_val = [], []  # 存储最佳结果
        # 遍历所有随机种子进行实验
        for seed in random_seeds:    
            print(f"current random seed: {seed}")

            # 根据训练模式选择数据文件
            if train_mode == "unified":
                # 统一模式：使用包含所有目标的数据集
                filename1 = '../data/raw_train_all.csv'
                filename2 = '../data/raw_val_all.csv'
                filename3 = '../data/raw_test_all.csv'
            elif train_mode == "adhoc":
                # 特定模式：使用针对特定目标的数据集
                filename1 = f'../data/raw_train_{target_word_pair[target_index]}.csv'
                filename2 = f'../data/raw_val_{target_word_pair[target_index]}.csv'
                filename3 = f'../data/raw_test_{target_word_pair[target_index]}.csv'
            
            # 加载并清理数据
            x_train, y_train, x_train_target = pp.clean_all(filename1, normalization_dict)
            x_val, y_val, x_val_target = pp.clean_all(filename2, normalization_dict)
            x_test, y_test, x_test_target = pp.clean_all(filename3, normalization_dict)
                
            # 确定标签数量
            num_labels = len(set(y_train))
            
            # 组织数据格式
            x_train_all = [x_train, y_train, x_train_target]
            x_val_all = [x_val, y_val, x_val_target]
            x_test_all = [x_test, y_test, x_test_target]
            
            # 设置随机种子以确保实验可重复性
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed) 
            # 确保CUDA操作也有确定性（如果使用GPU）
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True

            # 准备模型输入数据
            x_train_all, x_val_all, x_test_all = dh.data_helper_bert(x_train_all, x_val_all, x_test_all, model_select)
            
            # 创建数据加载器
            x_train_input_ids, x_train_seg_ids, x_train_atten_masks, y_train, x_train_len, trainloader = \
                                        dh.data_loader(x_train_all, batch_size, 'train')
            x_val_input_ids, x_val_seg_ids, x_val_atten_masks, y_val, x_val_len, valloader = \
                                        dh.data_loader(x_val_all, batch_size, 'val')                            
            x_test_input_ids, x_test_seg_ids, x_test_atten_masks, y_test, x_test_len, testloader = \
                                        dh.data_loader(x_test_all, batch_size, 'test')

            # 创建模型 - 移除.cuda()以确保在CPU上也能运行
            model = modeling.stance_classifier(num_labels, model_select)
            # 自动将模型移动到可用设备（CPU或GPU）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # 冻结BERT的嵌入层，只训练上层参数
            for n, p in model.named_parameters():
                if "bert.embeddings" in n:
                    p.requires_grad = False
            
            # 设置不同参数组的学习率
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')], 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')], 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
                ]
            
            # 定义损失函数和优化器
            loss_function = nn.CrossEntropyLoss(reduction='sum')  # 使用交叉熵损失
            optimizer = AdamW(optimizer_grouped_parameters)  # 使用AdamW优化器
            
            # 初始化结果记录列表
            sum_loss = []  # 训练损失记录
            val_f1_average = []  # 验证集平均F1分数记录
            
            # 根据训练模式初始化测试集F1分数记录
            if train_mode == "unified":
                test_f1_average = [[] for i in range(3)]  # 统一模式有3个子测试集
            elif train_mode == "adhoc":
                test_f1_average = [[]]  # 特定模式只有1个测试集

            # 训练循环
            for epoch in range(0, total_epoch):
                print(f'Epoch: {epoch}')
                train_loss = []  # 记录每个批次的损失
                
                # 设置模型为训练模式
                model.train()
                
                # 遍历训练数据
                for input_ids, seg_ids, atten_masks, target, length in trainloader:
                    # 梯度清零
                    optimizer.zero_grad()
                    # 前向传播
                    output1 = model(input_ids, seg_ids, atten_masks, length)
                    # 计算损失
                    loss = loss_function(output1, target)
                    # 反向传播
                    loss.backward()
                    # 梯度裁剪，防止梯度爆炸
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    # 参数更新
                    optimizer.step()
                    # 记录损失
                    train_loss.append(loss.item())
                
                # 计算平均损失
                sum_loss.append(sum(train_loss) / len(x_train))  
                print(sum_loss[epoch])

                # 在验证集上评估
                model.eval()
                val_preds = []
                # 不需要计算梯度
                with torch.no_grad():
                    for input_ids, seg_ids, atten_masks, target, length in valloader: 
                        pred1 = model(input_ids, seg_ids, atten_masks, length) 
                        val_preds.append(pred1)
                # 合并所有预测结果
                pred1 = torch.cat(val_preds, 0)
                # 计算评估指标
                acc, f1_average, precision, recall = model_eval.compute_f1(pred1, y_val)
                val_f1_average.append(f1_average)
                
                # 在测试集上评估
                with torch.no_grad():
                    test_preds = []
                    for input_ids, seg_ids, atten_masks, target, length in testloader:
                        pred1 = model(input_ids, seg_ids, atten_masks, length)
                        test_preds.append(pred1)
                    # 合并预测结果
                    pred1 = torch.cat(test_preds, 0)
                    
                    # 根据训练模式处理测试结果
                    if train_mode == "unified":
                        # 统一模式：分割测试集为三个子测试集
                        pred1_list = dh.sep_test_set(pred1)
                        y_test_list = dh.sep_test_set(y_test)
                    else:
                        # 特定模式：直接使用整个测试集
                        pred1_list = [pred1]
                        y_test_list = [y_test]
                        
                    # 对每个子测试集计算评估指标
                    for ind in range(len(y_test_list)):
                        pred1 = pred1_list[ind]
                        acc, f1_average, precision, recall = model_eval.compute_f1(pred1, y_test_list[ind])
                        test_f1_average[ind].append(f1_average)
            
            # 找到验证集上表现最好的轮次
            best_epoch = [index for index, v in enumerate(val_f1_average) if v == max(val_f1_average)][-1] 
            # 记录对应轮次的测试结果
            best_result.append([f1[best_epoch] for f1 in test_f1_average])

            # 打印验证集结果
            print("******************************************")
            print(f"dev results with seed {seed} on all epochs")
            print(val_f1_average)
            best_val.append(val_f1_average[best_epoch])
            
            # 打印测试集结果
            print("******************************************")
            print(f"test results with seed {seed} on all epochs")
            print(test_f1_average)
            print("******************************************")
        
        # 打印在验证集上表现最好的模型在测试集上的性能
        print("model performance on the test set: ")
        print(best_result)


if __name__ == "__main__":
    # 当直接运行脚本时执行主函数
    run_classifier()
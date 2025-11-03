import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


"""
计算模型的评估指标

参数:
    preds (torch.Tensor): 模型的原始预测输出
    y (torch.Tensor): 真实标签

返回:
    tuple: (准确率, 平均F1分数, 精确率, 召回率)
"""
def compute_f1(preds, y):
    
    # 应用softmax函数将输出转换为概率分布
    rounded_preds = F.softmax(preds)
    # 获取概率最大的类别索引作为预测结果
    _, indices = torch.max(rounded_preds, 1)
                
    # 计算准确率
    correct = (indices == y).float()
    acc = correct.sum() / len(correct)
    
    # 将张量转换为numpy数组用于计算其他指标
    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(y.cpu().numpy())
    
    # 计算精确率、召回率和F1分数
    # labels=[0,1] 表示我们关注的两个类别（反对和支持）
    result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
    
    # 计算支持类别和反对类别的平均F1分数
    f1_average = (result[2][0] + result[2][1]) / 2
        
    return acc, f1_average, result[0], result[1]  # 返回准确率、平均F1、精确率、召回率
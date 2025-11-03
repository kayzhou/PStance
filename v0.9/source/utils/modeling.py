import torch.nn as nn
# from transformers import AutoModel, BertModel
# 使用更安全的导入方式
import modelscope
from modelscope import AutoModel, AutoModelForSequenceClassification


"""
立场分类器模型 - 使用BERT或BERTweet作为基础模型
用于分类文本对目标实体的立场（支持或反对）
"""
class stance_classifier(nn.Module):
    
    """
    初始化立场分类器模型
    
    参数:
        num_labels (int): 分类标签的数量，通常为2（支持/反对）
        model_select (str): 选择使用的基础模型，'Bertweet'或'Bert'
    """
    def __init__(self, num_labels, model_select):
        # 调用父类初始化方法
        super(stance_classifier, self).__init__()
        
        # 定义网络层
        self.dropout = nn.Dropout(0.1)  # Dropout层，防止过拟合
        self.relu = nn.ReLU()  # ReLU激活函数
        self.tanh = nn.Tanh()  # Tanh激活函数
        
        # 根据选择加载预训练的BERT或BERTweet模型（使用modelscope）
        if model_select == 'Bertweet':
            # 使用本地缓存的模型kayzhou/bertweet-base
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                "kayzhou/bertweet-base",
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        elif model_select == 'Bert':
            # 使用modelscope的bert-base模型
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                "kayzhou/bert-base",
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        
        # 保存隐藏层大小信息
        self.hidden_size = 768  # BERT和BERTweet的标准隐藏层大小
        
        # 全连接层，用于特征转换
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        # 输出层，将特征映射到标签空间
        self.out = nn.Linear(self.hidden_size, num_labels)
        
    """
    前向传播函数
    
    参数:
        x_input_ids (torch.Tensor): 输入文本的token IDs
        x_seg_ids (torch.Tensor): 分段IDs，区分目标和文本
        x_atten_masks (torch.Tensor): 注意力掩码
        x_len (torch.Tensor): 输入序列长度
    
    返回:
        torch.Tensor: 模型的预测输出
    """
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len):
        # 调用BERT模型获取隐藏状态（适配modelscope API）
        outputs = self.bert(
            input_ids=x_input_ids, 
            attention_mask=x_atten_masks, 
            token_type_ids=x_seg_ids,
            return_dict=True
        )
        
        # 获取[CLS]标记的输出，用于分类
        # modelscope的分类模型可能输出logits或hidden_states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # 如果有hidden_states，使用最后一层的[CLS]标记
            query = outputs.hidden_states[-1][:, 0]  # [CLS]标记位于序列的第一个位置
        elif hasattr(outputs, 'logits'):
            # 如果直接有logits，使用它们
            return outputs.logits
        else:
            # 尝试从outputs中获取第一个元素作为最后一层隐藏状态
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            query = last_hidden[:, 0]  # [CLS]标记位于序列的第一个位置
        query = self.dropout(query)  # 应用dropout
        
        # 通过全连接层和激活函数
        linear = self.relu(self.linear(query))
        # 通过输出层
        out = self.out(linear)
        
        return out
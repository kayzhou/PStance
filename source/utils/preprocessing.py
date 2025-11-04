import preprocessor as p 
import re
import wordninja
import pandas as pd

# 配置 preprocessor 选项，只需设置一次
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)

# Data Loading
def load_data(filename):
    """加载数据并进行标签转换"""
    # 一次性读取CSV文件，提高效率
    data = pd.read_csv(filename, usecols=[0, 1, 2], encoding='ISO-8859-1')
    
    # 使用字典映射进行标签转换，更加清晰
    label_mapping = {'FAVOR': 1, 'NONE': 2, 'AGAINST': 0}
    data.iloc[:, 2] = data.iloc[:, 2].replace(label_mapping).infer_objects(copy=False)
    
    # 过滤掉标签为2的行
    filtered_data = data[data.iloc[:, 2] != 2]
    
    return filtered_data


# Data Cleaning
def data_clean(strings, norm_dict):
    """清理单条文本数据"""
    # 使用预配置的选项清理URL、表情等
    clean_data = p.clean(strings)  # 清理URL、emoji等
    
    # 移除特定标签
    clean_data = re.sub(r"#SemST", "", clean_data)
    
    # 分词处理
    tokens = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    
    # 规范化和分词处理
    result_tokens = []
    for token in tokens:
        token_lower = token.lower()
        # 检查是否在规范化字典中
        if token_lower in norm_dict:
            result_tokens.append(norm_dict[token_lower])
        # 处理hashtag和@提及
        elif token_lower.startswith("#") or token_lower.startswith("@"):
            result_tokens.extend(wordninja.split(token_lower))
        else:
            result_tokens.append(token_lower)
    
    return result_tokens


# Clean All Data
def clean_all(filename, norm_dict):
    """清理所有数据"""
    data = load_data(filename)
    
    # 获取列名（处理可能的列名变化）
    column_names = data.columns.tolist()
    tweet_col, target_col, stance_col = column_names[0], column_names[1], column_names[2]
    
    # 使用列表推导式替代循环，提高代码简洁性
    clean_tweets = [data_clean(text, norm_dict) for text in data[tweet_col].tolist()]
    clean_targets = [data_clean(target, norm_dict) for target in data[target_col].tolist()]
    labels = data[stance_col].tolist()
    
    return clean_tweets, labels, clean_targets


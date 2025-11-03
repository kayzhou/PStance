import preprocessor as p  # 用于清理社交媒体文本
import re  # 正则表达式库
import wordninja  # 用于拆分组合词（如hashtag）
import csv
import pandas as pd


"""
加载CSV格式的数据文件

参数:
    filename (str): CSV文件路径

返回:
    pandas.DataFrame: 包含文本、标签和目标实体的DataFrame
"""
def load_data(filename):
    # 将文件名转为列表格式
    filename = [filename]
    # 初始化DataFrame
    concat_text = pd.DataFrame()
    
    # 分别读取文本、标签和目标实体列
    # 使用ISO-8859-1编码以确保正确读取特殊字符
    raw_text = pd.read_csv(filename[0], usecols=[0], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename[0], usecols=[2], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename[0], usecols=[1], encoding='ISO-8859-1')
    
    # 将文本标签转换为数字标签
    # 'FAVOR' -> 1（支持）, 'NONE' -> 2（中立）, 'AGAINST' -> 0（反对）
    label = pd.DataFrame.replace(raw_label, ['FAVOR', 'NONE', 'AGAINST'], [1, 2, 0])
    
    # 合并文本、标签和目标实体
    concat_text = pd.concat([raw_text, label, raw_target], axis=1)
    
    # 过滤掉中立标签（Stance != 2），只保留支持和反对的样本
    concat_text = concat_text[concat_text.Stance != 2]
    
    return concat_text


"""
清理单个文本字符串

参数:
    strings (str): 待清理的文本
    norm_dict (dict): 规范化词典，用于替换缩写和特殊术语

返回:
    list: 清理后的单词列表
"""
def data_clean(strings, norm_dict):
    
    # 设置preprocessor选项，清理URL、表情符号和保留字符
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)
    # 使用preprocessor库清理文本
    clean_data = p.clean(strings)
    
    # 移除特定的标签标记
    clean_data = re.sub(r"#SemST", "", clean_data)
    
    # 使用正则表达式分割文本为单词、标点和数字
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/<>=$]|[0-9]+", clean_data)
    
    # 将每个单词转为小写并放入列表中
    clean_data = [[x.lower()] for x in clean_data]
    
    # 遍历每个单词进行规范化处理
    for i in range(len(clean_data)):
        # 如果单词在规范化词典中，进行替换
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i][0] = norm_dict[clean_data[i][0]]
            continue
        # 对于hashtag和提及(@)，使用wordninja进行拆分
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0])
    
    # 展平嵌套列表
    clean_data = [j for i in clean_data for j in i]

    return clean_data


"""
清理所有数据（文本和目标实体）

参数:
    filename (str): 数据文件路径
    norm_dict (dict): 规范化词典

返回:
    tuple: (清理后的文本列表, 标签列表, 清理后的目标实体列表)
"""
def clean_all(filename, norm_dict):
    
    # 加载数据
    concat_text = load_data(filename)
    
    # 提取原始文本、标签和目标实体
    raw_data = concat_text['Tweet'].values.tolist() 
    label = concat_text['Stance'].values.tolist()
    x_target = concat_text['Target'].values.tolist()
    
    # 初始化清理后的文本列表
    clean_data = [None for _ in range(len(raw_data))]
    
    # 对每个文本和目标实体进行清理
    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i], norm_dict)
        x_target[i] = data_clean(x_target[i], norm_dict)
    
    return clean_data, label, x_target


import os
import pandas as pd
import glob


def merge_csv_files():
    """
    合并data目录中的CSV文件：
    - 将raw_train_*.csv合并为raw_train_all.csv
    - 将raw_val_*.csv合并为raw_val_all.csv
    - 将raw_test_*.csv合并为raw_test_all.csv
    """
    # 获取当前脚本所在目录的父目录，然后拼接data目录路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # 定义要合并的文件类型
    file_types = {
        'train': 'raw_train_*.csv',
        'val': 'raw_val_*.csv',
        'test': 'raw_test_*.csv'
    }
    
    for file_type, pattern in file_types.items():
        # 构建完整的文件路径模式
        file_pattern = os.path.join(data_dir, pattern)
        # 获取所有匹配的文件
        csv_files = glob.glob(file_pattern)
        
        if not csv_files:
            print(f"未找到匹配 {pattern} 的文件")
            continue
        
        # 读取并合并所有CSV文件
        print(f"合并 {len(csv_files)} 个{file_type}文件...")
        df_list = []
        
        for file in csv_files:
            try:
                # 读取CSV文件
                df = pd.read_csv(file)
                # 添加源文件标识列（可选）
                # df['source_file'] = os.path.basename(file)
                df_list.append(df)
                print(f"  - 已读取: {os.path.basename(file)}")
            except Exception as e:
                print(f"  - 读取 {os.path.basename(file)} 时出错: {e}")
        
        # 合并所有DataFrame
        if df_list:
            merged_df = pd.concat(df_list, ignore_index=True)
            # 构建输出文件路径
            output_file = os.path.join(data_dir, f'raw_{file_type}_all.csv')
            # 保存合并后的文件
            merged_df.to_csv(output_file, index=False)
            print(f"已保存合并后的文件: {os.path.basename(output_file)}, "
                  f"共 {len(merged_df)} 行数据")
        else:
            print(f"没有成功读取任何{file_type}文件")



if __name__ == "__main__":
    print("开始合并CSV文件...")
    merge_csv_files()
    print("CSV文件合并完成！")
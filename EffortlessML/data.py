import pandas as pd


class MLData:

    def __init__(self, df: pd.DataFrame, x_cols: list[str], y_col: str):
        self.df = df
        self.x_cols = x_cols
        self.y_col = y_col

    def get_x(self):
        return self.df[self.x_cols]

    def get_y(self):
        return self.df[self.y_col]
    
    def report(self):
        print("数据集报告")
        print("=" * 30)
        
        # 1. 数据集总人数
        total_samples = len(self.df)
        print(f"1. 数据集总人数：{total_samples}")
        
        # 2. 特征数量
        num_features = len(self.x_cols)
        print(f"2. 特征数量：{num_features}")
        
        # 3. y_col的两个标签下的人数
        y_counts = self.df[self.y_col].value_counts()
        if len(y_counts) != 2:
            print(f"警告：y_col不是二元变量，共有 {len(y_counts)} 个不同的值")
        
        print("3. 目标变量的分布：")
        for label, count in y_counts.items():
            print(f"   标签 '{label}' 的样本数：{count}")
        
        print("=" * 30)

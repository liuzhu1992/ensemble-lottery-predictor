"""
数据加载与预处理模块
处理历史开奖记录的读取、验证、特征工程
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class LotteryDataLoader:
    """彩票历史数据加载器"""
    
    def __init__(self, csv_path: str):
        """
        初始化数据加载器
        
        Args:
            csv_path: 开奖记录CSV文件路径
        """
        self.csv_path = csv_path
        self.df = None
        self.red_balls = None
        self.blue_ball = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载CSV数据
        
        Returns:
            pd.DataFrame: 开奖记录数据框
        """
        self.df = pd.read_csv(self.csv_path)
        
        # 验证列名
        required_cols = ['开奖期号', '蓝', '红1', '红2', '红3', '红4', '红5', '红6']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"缺少必需列: {col}")
        
        # 数据类型转换
        self.df['开奖期号'] = self.df['开奖期号'].astype(str)
        for col in ['蓝', '红1', '红2', '红3', '红4', '红5', '红6']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 移除缺失值行
        self.df = self.df.dropna()
        
        # 排序
        self.df = self.df.reset_index(drop=True)
        
        print(f"✓ 加载数据成功: {len(self.df)} 期开奖记录")
        print(f"  期号范围: {self.df['开奖期号'].iloc[0]} - {self.df['开奖期号'].iloc[-1]}")
        
        return self.df
    
    def extract_balls(self):
        """提取红球和蓝球"""
        self.red_balls = self.df[['红1', '红2', '红3', '红4', '红5', '红6']].values
        self.blue_ball = self.df['蓝'].values
        
    def engineer_features(self) -> pd.DataFrame:
        """
        特征工程：衍生高阶时序特征
        
        Returns:
            pd.DataFrame: 增强型特征数据框
        """
        self.extract_balls()
        
        features = pd.DataFrame()
        features['period'] = range(len(self.df))  # 期次序号
        
        # 基础统计特征
        features['red_sum'] = self.red_balls.sum(axis=1)  # 红球和值
        features['blue_ball'] = self.blue_ball  # 蓝球
        
        # 奇偶特征
        features['red_odd_count'] = (self.red_balls % 2 == 1).sum(axis=1)  # 红球奇数个数
        features['red_even_count'] = 6 - features['red_odd_count']  # 红球偶数个数
        
        # 大小比特征 (17以上为大)
        features['red_big_count'] = (self.red_balls >= 17).sum(axis=1)
        features['red_small_count'] = 6 - features['red_big_count']
        
        # 区间分布 (1-11, 12-22, 23-33)
        features['red_zone1'] = ((self.red_balls >= 1) & (self.red_balls <= 11)).sum(axis=1)
        features['red_zone2'] = ((self.red_balls >= 12) & (self.red_balls <= 22)).sum(axis=1)
        features['red_zone3'] = ((self.red_balls >= 23) & (self.red_balls <= 33)).sum(axis=1)
        
        # 连号个数
        def count_consecutive(arr):
            if len(arr) < 2:
                return 0
            sorted_arr = np.sort(arr)
            consecutive = 0
            for i in range(len(sorted_arr) - 1):
                if sorted_arr[i+1] - sorted_arr[i] == 1:
                    consecutive += 1
            return consecutive
        
        features['red_consecutive'] = np.array([count_consecutive(row) for row in self.red_balls])
        
        # AC值 (号码复杂度)
        def calc_ac_value(arr):
            sorted_arr = np.sort(arr)
            ac = 0
            for i in range(len(sorted_arr) - 1):
                ac += sorted_arr[i+1] - sorted_arr[i] - 1
            return ac
        
        features['red_ac_value'] = np.array([calc_ac_value(row) for row in self.red_balls])
        
        # 号码跨度
        features['red_span'] = self.red_balls.max(axis=1) - self.red_balls.min(axis=1)
        
        # 冷热度特征 (最近20期频率)
        window = 20
        features['red_frequency'] = 0.0
        for i in range(len(self.df)):
            if i >= window:
                recent_balls = self.red_balls[i-window:i].flatten()
                current_balls = self.red_balls[i]
                freq = sum(1 for b in current_balls if b in recent_balls) / 6
                features.loc[i, 'red_frequency'] = freq
        
        # 遗漏值特征 (号码距离上次出现的期数)
        features['avg_miss'] = 0.0
        for i in range(len(self.df)):
            current_balls = self.red_balls[i]
            miss_list = []
            for ball in current_balls:
                # 查找该号码在历史中最后一次出现的位置
                last_appear = -1
                for j in range(i-1, -1, -1):
                    if ball in self.red_balls[j]:
                        last_appear = j
                        break
                miss_list.append(i - last_appear - 1)
            features.loc[i, 'avg_miss'] = np.mean(miss_list) if miss_list else i
        
        # 重号个数 (与上期相同的号码)
        features['repeat_with_last'] = 0
        for i in range(1, len(self.df)):
            repeat_count = len(set(self.red_balls[i]) & set(self.red_balls[i-1]))
            features.loc[i, 'repeat_with_last'] = repeat_count
        
        # 蓝球奇偶
        features['blue_odd'] = (self.blue_ball % 2).astype(int)
        
        print(f"✓ 特征工程完成: {features.shape[1]} 个特征")
        
        return features
    
    def get_train_test_split(self, test_periods: int = 30) -> Tuple[int, int]:
        """
        获取训练/测试分割点（时序交叉验证）
        
        Args:
            test_periods: 测试期数
            
        Returns:
            Tuple[int, int]: (train_end_index, test_start_index)
        """
        split_idx = len(self.df) - test_periods
        return split_idx, len(self.df)

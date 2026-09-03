"""
统计模型模块
包含冷热分析、贝叶斯估计、加权频率统计等传统统计方法
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter


class StatisticalModels:
    """统计基准模型集合"""
    
    def __init__(self, red_balls: np.ndarray, blue_ball: np.ndarray, alpha: float = 0.95):
        """
        初始化统计模型
        
        Args:
            red_balls: 红球开奖历史 (N, 6)
            blue_ball: 蓝球开奖历史 (N,)
            alpha: 时间衰减系数，越近期权重越高 [0.9, 0.99]
        """
        self.red_balls = red_balls
        self.blue_ball = blue_ball
        self.alpha = alpha
        self.n_periods = len(red_balls)
        
    def _get_decay_weights(self) -> np.ndarray:
        """
        生成时间衰减权重向量
        
        最近期的数据权重最高，历史数据权重衰减
        权重 w_i = alpha^(n - i - 1)
        
        Returns:
            np.ndarray: 权重向量，长度为 n_periods
        """
        weights = np.array([self.alpha ** (self.n_periods - i - 1) for i in range(self.n_periods)])
        weights = weights / weights.sum()  # 归一化
        return weights
    
    def cold_hot_analysis(self, red_only: bool = True) -> Dict:
        """
        冷热号码分析（加权频率统计）
        
        冷号：最近历史中出现频率低
        热号：最近历史中出现频率高
        
        Args:
            red_only: 仅分析红球(True)还是同时分析蓝球(False)
            
        Returns:
            Dict: 包含热号、冷号、频率等统计信息
        """
        weights = self._get_decay_weights()
        
        if red_only:
            # 红球分析 (1-33)
            frequency = np.zeros(34)  # 索引0不使用
            for period_idx, weight in enumerate(weights):
                for ball in self.red_balls[period_idx]:
                    frequency[int(ball)] += weight
            
            # 排序
            ball_freq = [(ball, freq) for ball, freq in enumerate(frequency[1:], 1)]
            ball_freq.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'hot_numbers': [b for b, _ in ball_freq[:10]],  # 前10个热号
                'cold_numbers': [b for b, _ in ball_freq[-10:]],  # 后10个冷号
                'frequencies': dict(ball_freq),
                'type': 'red_balls'
            }
        else:
            # 红球 + 蓝球
            red_freq = np.zeros(34)
            blue_freq = np.zeros(17)
            
            for period_idx, weight in enumerate(weights):
                for ball in self.red_balls[period_idx]:
                    red_freq[int(ball)] += weight
                blue_freq[int(self.blue_ball[period_idx])] += weight
            
            red_ball_freq = [(b, f) for b, f in enumerate(red_freq[1:], 1)]
            red_ball_freq.sort(key=lambda x: x[1], reverse=True)
            
            blue_ball_freq = [(b, f) for b, f in enumerate(blue_freq[1:], 1)]
            blue_ball_freq.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'hot_red': [b for b, _ in red_ball_freq[:10]],
                'cold_red': [b for b, _ in red_ball_freq[-10:]],
                'hot_blue': [b for b, _ in blue_ball_freq[:5]],
                'cold_blue': [b for b, _ in blue_ball_freq[-5:]],
                'red_freq': dict(red_ball_freq),
                'blue_freq': dict(blue_ball_freq)
            }
    
    def bayesian_estimation(self) -> Dict:
        """
        贝叶斯概率估计
        
        P(ball) = (出现次数 + 先验) / (总期数 + 先验调整项)
        使用加权计数
        
        Returns:
            Dict: 每个号码的后验概率估计
        """
        weights = self._get_decay_weights()
        weighted_sum = weights.sum()
        
        # 红球概率
        red_counts = np.zeros(34)
        for period_idx, weight in enumerate(weights):
            for ball in self.red_balls[period_idx]:
                red_counts[int(ball)] += weight
        
        # 加入拉普拉斯平滑 (先验 = 1)
        alpha_smooth = 1.0
        red_probs = (red_counts[1:] + alpha_smooth) / (weighted_sum + 33 * alpha_smooth)
        
        # 蓝球概率
        blue_counts = np.zeros(17)
        for period_idx, weight in enumerate(weights):
            blue_counts[int(self.blue_ball[period_idx])] += weight
        
        blue_probs = (blue_counts[1:] + alpha_smooth) / (weighted_sum + 16 * alpha_smooth)
        
        return {
            'red_probs': {ball: prob for ball, prob in enumerate(red_probs, 1)},
            'blue_probs': {ball: prob for ball, prob in enumerate(blue_probs, 1)},
            'red_expected_count': 6 * red_probs.sum() / 33  # 期望每期出现6个
        }
    
    def missing_value_analysis(self) -> Dict:
        """
        遗漏值分析：号码距离上次出现的期数
        
        遗漏期数长的号码下期出现概率相对增加（虽然理论上独立）
        这是从数据角度尝试挖掘的统计模式
        
        Returns:
            Dict: 每个号码的平均遗漏期数
        """
        missing_periods = {}
        current_period = self.n_periods
        
        for ball in range(1, 34):
            last_appear = -1
            miss_counts = []
            
            for period_idx in range(self.n_periods):
                if ball in self.red_balls[period_idx]:
                    if last_appear >= 0:
                        miss_counts.append(period_idx - last_appear)
                    last_appear = period_idx
            
            # 当前遗漏期数
            current_miss = current_period - last_appear - 1 if last_appear >= 0 else current_period
            avg_miss = np.mean(miss_counts) if miss_counts else current_miss
            
            missing_periods[ball] = {
                'current_miss': current_miss,
                'average_miss': avg_miss,
                'max_miss': max(miss_counts) if miss_counts else 0,
                'frequency': len(miss_counts)
            }
        
        return missing_periods
    
    def number_pair_correlation(self) -> Dict:
        """
        号码对相关性分析
        
        统计哪些号码经常一起出现
        
        Returns:
            Dict: 高频号码对及其出现次数
        """
        pair_counts = Counter()
        
        for period_idx in range(self.n_periods):
            balls = sorted(self.red_balls[period_idx])
            # 生成所有号码对
            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    pair = (int(balls[i]), int(balls[j]))
                    pair_counts[pair] += 1
        
        # 获取最频繁的号码对
        top_pairs = pair_counts.most_common(20)
        
        return {
            'top_pairs': top_pairs,
            'pair_stats': dict(pair_counts)
        }
    
    def range_distribution_tendency(self) -> Dict:
        """
        区间分布趋势分析（1-11, 12-22, 23-33）
        
        分析最近N期内三个区间的号码分布情况
        
        Returns:
            Dict: 各区间的平均号码数和趋势
        """
        weights = self._get_decay_weights()
        
        zone_counts = np.zeros(3)  # 三个区间
        for period_idx, weight in enumerate(weights):
            balls = self.red_balls[period_idx]
            
            zone1 = sum(1 for b in balls if 1 <= b <= 11)
            zone2 = sum(1 for b in balls if 12 <= b <= 22)
            zone3 = sum(1 for b in balls if 23 <= b <= 33)
            
            zone_counts[0] += zone1 * weight
            zone_counts[1] += zone2 * weight
            zone_counts[2] += zone3 * weight
        
        total_weight = weights.sum()
        avg_distribution = zone_counts / total_weight
        
        return {
            'zone1_avg': avg_distribution[0],  # 1-11
            'zone2_avg': avg_distribution[1],  # 12-22
            'zone3_avg': avg_distribution[2],  # 23-33
            'distribution': avg_distribution.tolist()
        }
    
    def predict_candidates(self, top_k: int = 15) -> Dict:
        """
        综合统计模型预测候选号码
        
        综合冷热、遗漏、相关性等多个因素
        
        Args:
            top_k: 返回前k个候选号码
            
        Returns:
            Dict: 红球候选号码及蓝球候选号码
        """
        # 冷热分析
        cold_hot = self.cold_hot_analysis(red_only=True)
        hot_numbers = cold_hot['hot_numbers']
        cold_numbers = cold_hot['cold_numbers']
        
        # 遗漏值分析
        missing = self.missing_value_analysis()
        # 遗漏期数长的号码
        missing_sorted = sorted(missing.items(), 
                               key=lambda x: x[1]['current_miss'], 
                               reverse=True)[:10]
        missing_candidates = [ball for ball, _ in missing_sorted]
        
        # 贝叶斯概率
        bayes = self.bayesian_estimation()
        red_probs = bayes['red_probs']
        blue_probs = bayes['blue_probs']
        
        # 综合评分：热号权重高 + 适度遗漏权重
        red_scores = {}
        for ball in range(1, 34):
            score = 0
            
            # 热号倾向
            if ball in hot_numbers:
                score += 0.3
            if ball in missing_candidates:
                score += 0.2
            
            # 概率权重
            score += 0.5 * red_probs[ball]
            
            red_scores[ball] = score
        
        # 排序获取候选号码
        red_candidates = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        red_pred = [b for b, _ in red_candidates[:top_k]]
        
        # 蓝球预测
        blue_candidates = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)
        blue_pred = [b for b, _ in blue_candidates[:8]]
        
        return {
            'red_candidates': red_pred,
            'blue_candidates': blue_pred,
            'red_scores': red_scores,
            'blue_probs': blue_probs
        }


class ProbabilityCalculator:
    """概率计算工具"""
    
    @staticmethod
    def joint_probability(red_probs: Dict, selected_reds: List[int]) -> float:
        """
        计算选定红球的联合概率（假设独立）
        
        P(selected_reds) = P(r1) * P(r2) * ... * P(r6) 的近似
        实际上是概率的几何平均
        
        Args:
            red_probs: 单个红球概率字典
            selected_reds: 选定的红球号码列表
            
        Returns:
            float: 联合概率估计
        """
        if len(selected_reds) != 6:
            raise ValueError("必须选择6个红球")
        
        probs = [red_probs[ball] for ball in selected_reds]
        # 使用几何平均作为近似
        joint_prob = np.prod(probs) ** (1/6)
        return joint_prob
    
    @staticmethod
    def combination_score(red_numbers: List[int], 
                         red_scores: Dict, 
                         blue_number: int,
                         blue_probs: Dict) -> float:
        """
        计算投注组合的综合概率得分
        
        Args:
            red_numbers: 红球号码列表
            red_scores: 红球得分字典
            blue_number: 蓝球号码
            blue_probs: 蓝球概率字典
            
        Returns:
            float: 组合总得分
        """
        red_score = sum(red_scores.get(b, 0) for b in red_numbers) / len(red_numbers)
        blue_score = blue_probs.get(blue_number, 0)
        
        # 加权组合 (红球权重70%，蓝球权重30%)
        total_score = 0.7 * red_score + 0.3 * blue_score
        return total_score

# statistical_analyzer.py - 统计分析方法
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Any
import json


class StatisticalAnalyzer:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def _convert_to_serializable(self, obj):
        """将 NumPy 类型转换为 Python 原生类型以便 JSON 序列化"""
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (dict,)):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def calculate_confidence_interval(self, data: List[float]) -> Dict[str, float]:
        """计算置信区间"""
        if len(data) < 2:
            result = {"mean": np.mean(data), "ci_lower": np.mean(data), "ci_upper": np.mean(data)}
        else:
            mean = np.mean(data)
            sem = stats.sem(data)
            ci = stats.t.interval(self.confidence_level, len(data) - 1, loc=mean, scale=sem)

            result = {
                "mean": mean,
                "std": np.std(data),
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "sample_size": len(data)
            }

        # 转换为可序列化的类型
        return self._convert_to_serializable(result)

    def anova_analysis(self, data_groups: Dict[str, List[float]]) -> Dict[str, Any]:
        """方差分析比较不同实验条件下的差异"""
        try:
            f_stat, p_value = stats.f_oneway(*data_groups.values())
            result = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "group_means": {k: np.mean(v) for k, v in data_groups.items()}
            }
        except Exception as e:
            result = {"error": str(e)}

        # 转换为可序列化的类型
        return self._convert_to_serializable(result)

    def effect_size_analysis(self, data1: List[float], data2: List[float]) -> Dict[str, float]:
        """计算效应量（Cohen's d）"""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)

        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std

        result = {
            "cohens_d": cohens_d,
            "effect_size": "large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small"
        }

        # 转换为可序列化的类型
        return self._convert_to_serializable(result)
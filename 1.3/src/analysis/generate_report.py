# visualization/generate_report.py - 加入 GWF 报告
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from config import REPORT_DIR


def generate_report():
    jsonl_path = REPORT_DIR / "energy_summary.jsonl"

    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        print("错误: energy_summary.jsonl 文件不存在或为空！请先运行实验。")
        return

    # 读取所有测量记录
    df = pd.read_json(jsonl_path, lines=True)

    if df.empty:
        print("警告: energy_summary.jsonl 数据为空！")
        return

    # ==================== 关键修复：使用净能耗 ====================
    df = df[~df['phase'].str.startswith('baseline', na=False)].copy()

    if df.empty:
        print("警告: 排除 baseline 后无有效操作数据！")
        return

    # 优先使用已计算的 net_energy_J（新实验会保存）
    if 'net_energy_J' in df.columns:
        energy_col = 'net_energy_J'
        print(f"[Report] 使用已保存的净能耗字段 '{energy_col}'")
    else:
        # 旧数据兼容：重新计算净能耗（使用典型基线功率 42W，或从 baseline 行推算）
        print("[Report] 未找到 net_energy_J 字段，正在自动重新计算净能耗...")
        baseline_df = pd.read_json(jsonl_path, lines=True)
        baseline_rows = baseline_df[baseline_df['phase'].str.startswith('baseline', na=False)]
        if not baseline_rows.empty and 'duration_s' in baseline_rows.columns:
            total_baseline_energy = baseline_rows['energy_J'].sum()
            total_baseline_duration = baseline_rows['duration_s'].sum()
            baseline_power = total_baseline_energy / total_baseline_duration if total_baseline_duration > 0 else 42.0
        else:
            baseline_power = 42.0

        df['net_energy_J'] = df['energy_J'] - baseline_power * df['duration_s']
        df['net_energy_J'] = df['net_energy_J'].clip(lower=0.0)
        energy_col = 'net_energy_J'
        print(f"[Report] 使用重新计算的净能耗（基线功率 ≈ {baseline_power:.2f} W）")

    # ==================== 聚合统计，同时统计 workload_factor ====================
    agg_dict = {
        energy_col: ['mean', 'std', 'count']
    }
    if 'workload_factor' in df.columns:
        agg_dict['workload_factor'] = 'mean'

    summary = df.groupby('phase').agg(agg_dict)
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    # 计算 95% 置信区间
    summary['sem'] = summary[f'{energy_col}_std'] / np.sqrt(summary[f'{energy_col}_count'])
    summary['ci'] = summary['sem'] * stats.t.ppf((1 + 0.95) / 2., np.maximum(summary[f'{energy_col}_count'] - 1, 1))

    summary = summary.dropna(subset=[f'{energy_col}_mean', 'ci']).reset_index(drop=True)

    if summary.empty:
        print("错误: 聚合后无有效数据！")
        return

    # 排序
    summary = summary.sort_values(f'{energy_col}_mean', ascending=False).reset_index(drop=True)

    # ==================== 打印 & 保存 ====================
    if 'workload_factor_mean' in summary.columns:
        print("\n平均 GWF: (按 phase)")
        print(summary[['phase', 'workload_factor_mean']])

    output_csv = REPORT_DIR / "final_results_with_gwf.csv"
    summary.to_csv(output_csv, index=False)
    print(f"[Report] 已保存包含 GWF 的结果: {output_csv}")

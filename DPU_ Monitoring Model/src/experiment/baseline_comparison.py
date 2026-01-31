# baseline_comparison.py - 修复版：适配 measure_start / measure_stop 接口
import sys
import time
from pathlib import Path
import numpy as np
from scipy import stats

# 添加项目路径以支持导入
project_root = Path(__file__).parent.parent.parent
src_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.cypher_executor import CypherExecutor
from monitoring.energy_monitor_libre import EnergyMonitorLibre

class BaselineComparison:
    def __init__(self, executor: CypherExecutor, monitor: EnergyMonitorLibre):
        self.executor = executor
        self.monitor = monitor

    def baseline_power_consumption(self, duration: float = 300.0, repeats: int = 5) -> dict:
        print("[Baseline] Measuring stable baseline power consumption...")
        
        # 修复：在开始测量前，先确保数据库状态干净（如果启用清理）
        # 基线测量时不需要清理数据库，因为基线测量只是测量系统闲置功率，不涉及数据库操作
        # 清理数据库会消耗大量时间，影响基线测量的准确性
        print("[Baseline] 基线测量不需要清理数据库（仅测量系统闲置功率）")

        energies = []
        for i in range(repeats):
            print(f"  Baseline run {i + 1}/{repeats} (duration: {duration}s)")

            phase = f"baseline_idle_run{i}"
            self.monitor.measure_start(phase)
            
            # 在这个循环内部，只进行闲置等待，不执行任何数据库操作
            time.sleep(duration)
            
            result = self.monitor.measure_stop(workload_factor=0.0)

            energy = result["energy_J"]
            energies.append(energy)
            print(f"    → Run {i + 1} energy: {energy:.2f} J")

        mean_energy = np.mean(energies)
        sem = stats.sem(energies)
        ci = sem * stats.t.ppf((1 + 0.95) / 2., len(energies) - 1) if len(energies) > 1 else 0

        # 关键修复：正确计算平均功率（W = J/s）
        baseline_power_W = mean_energy / duration
        baseline_power_ci_W = ci / duration

        print(f"[Baseline] 平均闲置能耗: {mean_energy:.2f} ± {ci:.2f} J over {duration}s")
        print(f"[Baseline] 平均闲置功率: {baseline_power_W:.2f} ± {baseline_power_ci_W:.2f} W")

        return {
            "baseline_energy_J": mean_energy,
            "baseline_ci_J": ci,
            "baseline_power_W": baseline_power_W,  # 正确功率（W）
            "baseline_power_ci_W": baseline_power_ci_W,
            "samples": repeats
        }

    def net_energy_consumption(self, gross_energy: float, baseline_energy: float) -> dict:
        net_energy = max(0, gross_energy - baseline_energy)
        relative_overhead = (net_energy / baseline_energy) * 100 if baseline_energy > 0 else 0
        return {
            "gross_energy_J": gross_energy,
            "baseline_energy_J": baseline_energy,
            "net_energy_J": net_energy,
            "relative_overhead_percent": relative_overhead
        }
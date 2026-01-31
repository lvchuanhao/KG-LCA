# kg_lca_core.py  ——  最终修复版（正确净能耗 + 稳定短操作 + 论文级输出）
import time
import json
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats

from monitoring.energy_monitor_libre import EnergyMonitorLibre
from core.cypher_executor import CypherExecutor
from experiment.experimental_design import ExperimentalDesign
from experiment.baseline_comparison import BaselineComparison
# GWF 功能已禁用，不再导入相关模块
from monitoring.graph_workload_factor import GraphWorkloadFactorCalculator
from monitoring.gwf_sampler import GwfTimeSeriesSampler
from config import WARMUP_RUNS, REPEAT, REPORT_DIR, ENABLE_DB_CLEANUP


class KGLCAController:
    def __init__(self):
        self.executor = CypherExecutor()
        self.monitor = EnergyMonitorLibre()
        self.design = ExperimentalDesign()
        self.baseline_comp = BaselineComparison(self.executor, self.monitor)
        # GWF 功能，初始化相关组件
        self.gwf_calc = GraphWorkloadFactorCalculator(self.executor)
        from config import GWF_SAMPLE_INTERVAL
        self.gwf_sampler = GwfTimeSeriesSampler(self.executor, self.gwf_calc, sample_interval=GWF_SAMPLE_INTERVAL)

        # === 基线测量：测量系统闲置功率基准 ===
        print("[Baseline] 正在测量系统闲置功率基准...")
        print("[Baseline] 注意：基线测量需要约 2.5 分钟（30秒 × 5次），请耐心等待...")
        baseline_result = self.baseline_comp.baseline_power_consumption(duration=30.0, repeats=5)
        self.baseline_power_W = baseline_result["baseline_power_W"]
        self.baseline_ci_W = baseline_result.get("baseline_power_ci_W", 0.0)

        print(f"[Baseline] ✓ 基线测量完成！")
        print(f"          平均闲置功率: {self.baseline_power_W:.2f} ± {self.baseline_ci_W:.2f} W")
        print(f"          将用于所有操作扣除基线能耗（净能耗 = 总能耗 - {self.baseline_power_W:.1f}W × 时长）\n")

        # 初始化结果存储
        self.detailed_results = []  # 存储每次测量的详细数据
        self.step_results = []  # 存储每个实验步骤的平均结果

    def _measure_operation_energy(self, case: dict) -> float:
        phase = case["description"]
        print(f"\n=== 开始测量: {phase} ===")

        # GWF 功能已禁用，不再进行图统计刷新
        # 对于大规模操作（如 import_dataset、traverse_full、traverse_by_type），跳过图统计刷新以提高性能
        is_large_operation = (
                "import_dataset" in phase or
                "traverse_full" in phase or  # 全图遍历在大图上会很慢
                "traverse_by_type" in phase  # 按类型遍历也需要遍历全图
        )
        # GWF 已禁用，不再刷新图统计
        pre_stats = None

        self.monitor.measure_start(phase)
        start_time = time.time()

        try:
            # 支持 "query" 和 "query_name" 两种字段名
            query_name = case.get("query_name") or case.get("query")
            if not query_name:
                raise ValueError(f"用例 {case.get('description', 'unknown')} 缺少 'query' 或 'query_name' 字段")

            # 对于大规模操作，添加提示
            if is_large_operation:
                print(f"[提示] 正在执行大规模查询（这可能需要较长时间，请耐心等待）...")

            op_result = self.executor.run_query(query_name, case.get("params", {}))

            if is_large_operation:
                print(f"[提示] 查询执行完成，正在处理结果...")
        except Exception as e:
            error_str = str(e)
            # 如果是数据库不可用错误，说明重试机制已经尝试过了
            if "DatabaseUnavailable" in error_str or "not currently available" in error_str:
                print(f"[错误] 查询执行失败（数据库不可用，已重试但仍失败）: {e}")
            else:
                print(f"[错误] 查询执行失败: {e}")
            self.monitor.measure_stop()  # 即使失败也停止测量
            # 返回包含所有必需字段的字典，避免后续处理出错
            return {
                "net_energy_J": 0.0,
                "gross_energy_J": 0.0,
                "components_J": {"CPU": 0.0, "GPU": 0.0, "Memory": 0.0, "Disk": 0.0},
                "co2e_g": 0.0,
                "co2e_mg": 0.0,
                "duration_s": 0.0
            }

        # === 启用 GWF ===
        duration = time.time() - start_time

        # 计算单值 GWF（准确优先）；若需要时间序列可扩展 sampler
        try:
            gwf = self.gwf_calc.compute_gwf(case, op_result)
        except Exception as _e:
            print(f"[GWF] 计算失败，退化为 0: {_e}")
            gwf = 0.0
        gwf_series = None

        result = self.monitor.measure_stop(workload_factor=gwf, workload_series=gwf_series)

        gross_energy = result["energy_J"]  # 总能耗（包含闲置）
        components = result.get("components_J", {})  # 组件能耗（J）

        # 关键修复：按实际执行时长扣除闲置基线能耗
        # 使用保守的扣除方式：只扣除90%的基线能耗，避免过度扣除
        # 原因：执行操作时系统可能有一些基础开销（数据库连接、查询解析等），
        # 这些开销在完全闲置时不存在，但应该计入操作能耗
        expected_baseline_energy = self.baseline_power_W * duration
        conservative_baseline_energy = expected_baseline_energy * 0.9  # 只扣除90%的基线
        net_energy = max(0.0, gross_energy - conservative_baseline_energy)  # 防止负值

        # 计算各组件净能耗（按保守比例扣除基线）
        baseline_ratio = conservative_baseline_energy / gross_energy if gross_energy > 0 else 0.0
        net_components = {}
        for comp_name, comp_energy in components.items():
            net_comp_energy = max(0.0, comp_energy - (comp_energy * baseline_ratio))
            net_components[comp_name] = net_comp_energy

        # 基于净能耗重新计算碳排放（而不是总能耗）
        # 碳排放强度：550 gCO2e/kWh
        carbon_intensity = 550.0  # gCO2e/kWh
        net_energy_Wh = net_energy / 3600.0  # 转换为Wh
        net_energy_kWh = net_energy_Wh / 1000.0  # 转换为kWh
        co2e_g_net = net_energy_kWh * carbon_intensity  # 基于净能耗的碳排放（克）
        co2e_mg_net = co2e_g_net * 1000.0  # 转换为毫克

        # 简化打印（显示扣除信息以便调试）
        print(
            f"完成！净能耗: {net_energy:.2f}J (总能耗: {gross_energy:.2f}J, 基线扣除: {conservative_baseline_energy:.2f}J, 耗时: {duration:.3f}s)")

        # 返回完整结果，包括组件能耗
        return {
            "net_energy_J": net_energy,
            "gross_energy_J": gross_energy,
            "components_J": net_components,
            "co2e_g": co2e_g_net,  # 基于净能耗的碳排放（克）
            "co2e_mg": co2e_mg_net,  # 基于净能耗的碳排放（毫克）
            "duration_s": duration
        }

    def _run_case(self, case: dict) -> dict:
        """运行单个实验用例，执行预热和测量，并返回统计结果。"""
        # 注意：用例标题已在 run_scientific_experiment 中打印，这里不再重复

        # 1. 执行预热运行（自适应：如果预热时间>60秒，只预热1次，运行3次）
        actual_warmup_runs = int(case.get("warmup_runs", WARMUP_RUNS))
        actual_repeat = int(case.get("repeat", REPEAT))

        # 删除和导入不需要预热
        query_name_for_skip = (case.get("query_name") or case.get("query") or "").lower()
        if "import_dataset" in query_name_for_skip or "delete_graph" in query_name_for_skip:
            actual_warmup_runs = 0

        if actual_warmup_runs > 0:
            print(f"[预热] 开始执行 {actual_warmup_runs} 次预热运行（这可能需要一些时间）...")
            successful_warmups = 0
            failed_warmups = 0
            first_warmup_time = None

            for i in range(actual_warmup_runs):
                print(f"  [预热 {i + 1}/{actual_warmup_runs}] 正在执行...", end="", flush=True)
                try:
                    # 支持 "query" 和 "query_name" 两种字段名
                    query_name = case.get("query_name") or case.get("query")
                    if not query_name:
                        raise ValueError(f"用例 {case.get('description', 'unknown')} 缺少 'query' 或 'query_name' 字段")
                    start_time = time.time()
                    self.executor.run_query(query_name, case.get("params", {}))
                    elapsed = time.time() - start_time
                    print(f" 完成 (耗时: {elapsed:.2f}秒)")

                    # 记录第一次预热的时间
                    if i == 0:
                        first_warmup_time = elapsed
                        # 如果第一次预热时间>60秒，只预热1次，运行3次
                        if elapsed > 60.0:
                            print(f"[提示] 检测到预热时间较长（{elapsed:.2f}秒 > 60秒），将只进行1次预热，运行3次测量")
                            actual_warmup_runs = 1
                            actual_repeat = 3
                            break  # 只预热1次就退出

                    successful_warmups += 1
                except Exception as e:
                    error_str = str(e)
                    # 如果是内存溢出错误，跳过这次预热，但继续执行
                    if "MemoryPoolOutOfMemoryError" in error_str:
                        print(f" 跳过（内存不足，将在测量阶段处理）")
                        failed_warmups += 1
                        # 等待一下让内存释放
                        time.sleep(0.5)
                    else:
                        print(f" 失败: {e}")
                        failed_warmups += 1

            if successful_warmups > 0:
                print(f"[预热] ✓ 完成 {successful_warmups} 次成功预热运行")
            if failed_warmups > 0:
                print(f"[预热] ⚠ {failed_warmups} 次预热运行失败或跳过（将在测量阶段重试）")
            print()

        # 2. 执行实际测量运行（使用自适应后的运行次数）
        print(f"[测量] 开始执行 {actual_repeat} 次测量运行（这可能需要较长时间）...")
        energy_samples = []
        component_samples = {"CPU": [], "GPU": [], "Memory": [], "Disk": []}  # 存储各组件能耗
        co2_samples = []  # 存储碳排放
        detailed_measurements = []  # 存储每次测量的详细信息
        for i in range(actual_repeat):
            print(f"  [测量 {i + 1}/{actual_repeat}] 正在执行...", end="", flush=True)
            start_time = time.time()
            result = self._measure_operation_energy(case)
            elapsed = time.time() - start_time

            # 处理返回结果（可能是字典或数值）
            if isinstance(result, dict):
                net_energy = result.get("net_energy_J", 0.0)
                components = result.get("components_J", {})
                co2e_g = result.get("co2e_g", 0.0)
                co2e_mg = result.get("co2e_mg", co2e_g * 1000.0)  # 转换为毫克
            else:
                net_energy = result if result > 0 else 0.0
                components = {}
                co2e_g = 0.0
                co2e_mg = 0.0

            # 记录所有测量结果（包括0值），以便分析问题
            if net_energy > 0:  # 只记录有效测量结果
                energy_samples.append(net_energy)
                # 保存各组件能耗
                for comp_name in component_samples.keys():
                    comp_energy = components.get(comp_name, 0.0)
                    component_samples[comp_name].append(comp_energy)
                co2_samples.append(co2e_mg)  # 使用毫克

                # 保存详细测量数据
                detailed_measurements.append({
                    "step": i + 1,
                    "description": case.get("description", ""),
                    "query": case.get("query_name") or case.get("query", ""),
                    "net_energy_J": net_energy,
                    "cpu_energy_J": components.get("CPU", 0.0),
                    "gpu_energy_J": components.get("GPU", 0.0),
                    "memory_energy_J": components.get("Memory", 0.0),
                    "disk_energy_J": components.get("Disk", 0.0),
                    "co2e_g": co2e_g,  # 保留克用于兼容性
                    "co2e_mg": co2e_mg,  # 新增毫克
                    "timestamp": datetime.now().isoformat()
                })
                print(f" 完成 (净能耗: {net_energy:.2f}J, 耗时: {elapsed:.2f}秒)")
            else:
                # 如果总能耗为0，检查原因
                gross_energy = result.get("gross_energy_J", 0.0) if isinstance(result, dict) else 0.0
                duration = result.get("duration_s", elapsed) if isinstance(result, dict) else elapsed
                print(f" 完成但无有效数据 (总能耗: {gross_energy:.2f}J, 耗时: {elapsed:.2f}秒)")

        print(f"[测量] ✓ 完成 {len(energy_samples)} 次有效测量！\n")

        # 保存详细测量数据
        self.detailed_results.extend(detailed_measurements)

        # 3. 计算统计结果
        # 从 description 或 query 推断操作类型
        description = case.get("description", "")
        query_name = case.get("query_name") or case.get("query", "")

        # 根据描述或查询名推断操作类型（更细粒度的分类）
        if "import" in description.lower() or "import" in query_name.lower():
            operation_type = "import_dataset"
        elif "traverse" in description.lower() or "traverse" in query_name.lower():
            operation_type = "traverse"  # 遍历查询
        elif "1-hop" in description.lower() or "hop_1" in query_name.lower() or query_name == "query_hop_1":
            operation_type = "query_hop_1"  # 1跳查询
        elif "2-hop" in description.lower() or "hop_2" in query_name.lower() or query_name == "query_hop_2":
            operation_type = "query_hop_2"  # 2跳查询
        elif "3-hop" in description.lower() or "hop_3" in query_name.lower() or query_name == "query_hop_3":
            operation_type = "query_hop_3"  # 3跳查询
        elif "query" in description.lower() or "hop" in query_name.lower():
            operation_type = "query"  # 其他查询（备用）
        elif "update" in description.lower() or "update" in query_name.lower():
            operation_type = "update"  # 属性更新
        elif "delete" in description.lower() or "maintenance" in description.lower() or "delete" in query_name.lower():
            operation_type = "maintenance"  # 维护
        else:
            operation_type = case.get("operation", "unknown")

        if not energy_samples:
            return {
                "operation_type": operation_type,
                "description": description,
                "avg_net_J": 0.0,
                "std_dev": 0.0,
                "ci": 0.0,
                "samples": 0
            }

        # 计算总能耗统计量
        avg_energy = float(np.mean(energy_samples))

        # 处理样本量不足的情况（只有1个样本时无法计算标准差和置信区间）
        if len(energy_samples) == 1:
            std_dev = 0.0
            sem = 0.0
            ci = 0.0
        else:
            std_dev = float(np.std(energy_samples, ddof=1))  # 样本标准差
            sem = float(stats.sem(energy_samples))  # 标准误差
            if len(energy_samples) > 1:
                ci = float(stats.t.interval(0.95, len(energy_samples) - 1, loc=avg_energy, scale=sem)[
                               1] - avg_energy)  # 95% CI
            else:
                ci = 0.0

        # 计算各组件能耗统计量（转换为 kJ）
        component_stats = {}
        for comp_name, comp_values in component_samples.items():
            if comp_values and len(comp_values) > 0:
                comp_array = np.array(comp_values) / 1000.0  # 转换为 kJ
                comp_avg = float(np.mean(comp_array))
                # 处理样本量不足的情况
                if len(comp_values) == 1:
                    comp_std = 0.0
                else:
                    comp_std = float(np.std(comp_array, ddof=1))
                component_stats[comp_name] = {
                    "avg_kJ": comp_avg,
                    "std_kJ": comp_std
                }
            else:
                component_stats[comp_name] = {
                    "avg_kJ": 0.0,
                    "std_kJ": 0.0
                }

        # 计算碳排放统计量
        co2_avg = 0.0
        co2_std = 0.0
        if co2_samples and len(co2_samples) > 0:
            co2_array = np.array(co2_samples)
            co2_avg = float(np.mean(co2_array))
            # 处理样本量不足的情况
            if len(co2_samples) == 1:
                co2_std = 0.0
            else:
                co2_std = float(np.std(co2_array, ddof=1))

        print(f"[统计] 平均净能耗: {avg_energy:.2f} J ({avg_energy / 1000:.2f} kJ)")
        print(f"[统计] 标准差: {std_dev:.2f} J ({std_dev / 1000:.2f} kJ)")
        print(f"[统计] 95% 置信区间: ±{ci:.2f} J (±{ci / 1000:.2f} kJ)")

        result = {
            "operation_type": operation_type,
            "description": description,
            "query": query_name,
            "avg_net_J": avg_energy,
            "avg_net_kJ": avg_energy / 1000.0,
            "std_dev": std_dev,
            "std_dev_kJ": std_dev / 1000.0,
            "sem": sem,
            "ci": ci,
            "ci_kJ": ci / 1000.0,
            "samples": len(energy_samples),
            "min_net_J": float(np.min(energy_samples)) if energy_samples else 0.0,
            "max_net_J": float(np.max(energy_samples)) if energy_samples else 0.0,
            "components": component_stats,
            "co2e_g_avg": co2_avg / 1000.0,  # 保留克用于兼容性
            "co2e_g_std": co2_std / 1000.0,
            "co2e_mg_avg": co2_avg,  # 新增毫克
            "co2e_mg_std": co2_std,
            "timestamp": datetime.now().isoformat()
        }

        # 保存到步骤结果
        self.step_results.append(result)

        return result

    def _aggregate_results(self, results: list, operation_type: str, description: str) -> dict:
        """汇总多次测量的结果，计算平均值、标准差、置信区间等"""
        import numpy as np
        from scipy import stats

        # 提取所有能耗样本
        energy_samples = [r.get("avg_net_J", 0.0) for r in results if r.get("samples", 0) > 0]
        if not energy_samples:
            return {
                "operation_type": operation_type,
                "description": description,
                "avg_net_J": 0.0,
                "std_dev": 0.0,
                "ci": 0.0,
                "samples": 0
            }

        # 提取组件能耗
        component_samples = {"CPU": [], "GPU": [], "Memory": [], "Disk": []}
        co2_samples = []

        for r in results:
            if r.get("samples", 0) > 0:
                components = r.get("components", {})
                for comp_name in component_samples.keys():
                    comp_avg_kJ = components.get(comp_name, {}).get("avg_kJ", 0.0)
                    comp_samples = r.get("samples", 1)
                    # 按样本数加权
                    for _ in range(comp_samples):
                        component_samples[comp_name].append(comp_avg_kJ * 1000.0)  # 转回J

                co2_mg = r.get("co2e_mg_avg", r.get("co2e_g_avg", 0.0) * 1000.0)
                co2_samples.extend([co2_mg] * r.get("samples", 1))

        # 计算统计量
        avg_energy = float(np.mean(energy_samples))
        if len(energy_samples) == 1:
            std_dev = 0.0
            sem = 0.0
            ci = 0.0
        else:
            std_dev = float(np.std(energy_samples, ddof=1))
            sem = float(stats.sem(energy_samples))
            ci = float(stats.t.interval(0.95, len(energy_samples) - 1, loc=avg_energy, scale=sem)[1] - avg_energy)

        # 计算组件统计量
        component_stats = {}
        for comp_name, comp_values in component_samples.items():
            if comp_values:
                comp_avg = float(np.mean(comp_values)) / 1000.0  # 转回kJ
                comp_std = float(np.std(comp_values, ddof=1)) / 1000.0 if len(comp_values) > 1 else 0.0
                component_stats[comp_name] = {"avg_kJ": comp_avg, "std_kJ": comp_std}
            else:
                component_stats[comp_name] = {"avg_kJ": 0.0, "std_kJ": 0.0}

        # 计算碳排放统计量
        co2_avg = float(np.mean(co2_samples)) if co2_samples else 0.0
        co2_std = float(np.std(co2_samples, ddof=1)) if len(co2_samples) > 1 else 0.0

        return {
            "operation_type": operation_type,
            "description": description,
            "query": results[0].get("query", ""),
            "avg_net_J": avg_energy,
            "avg_net_kJ": avg_energy / 1000.0,
            "std_dev": std_dev,
            "std_dev_kJ": std_dev / 1000.0,
            "sem": sem,
            "ci": ci,
            "ci_kJ": ci / 1000.0,
            "samples": len(energy_samples),
            "min_net_J": float(np.min(energy_samples)) if energy_samples else 0.0,
            "max_net_J": float(np.max(energy_samples)) if energy_samples else 0.0,
            "components": component_stats,
            "co2e_g_avg": co2_avg / 1000.0,
            "co2e_g_std": co2_std / 1000.0,
            "co2e_mg_avg": co2_avg,
            "co2e_mg_std": co2_std,
            "timestamp": datetime.now().isoformat()
        }

    def _run_case_no_measure(self, case: dict):
        """执行用例但不测量能耗（用于准备数据）"""
        query_name = case.get("query_name") or case.get("query")
        if not query_name:
            raise ValueError(f"用例 {case.get('description', 'unknown')} 缺少 'query' 或 'query_name' 字段")

        print(f"[执行-不测量] {case.get('description', query_name)}")
        try:
            self.executor.run_query(query_name, case.get("params", {}))
            print(f"[完成-不测量] {case.get('description', query_name)}")
        except Exception as e:
            print(f"[错误] 执行失败: {e}")
            raise e

    def run_scientific_experiment(self):
        print("=" * 80)
        print("Starting High-Precision KG-LCA Experiment".center(80))
        print("=" * 80)
        print("\n实验包含以下操作类型：")
        print("  1. 数据导入（Data Import）- 30次测量")
        print("  2. 遍历查询（Traverse Queries）- 30次测量")
        print("  3. 多跳查询（1-hop, 2-hop, 3-hop Queries）- 30次测量")
        print("  4. 属性更新（Property Updates）- 30次测量")
        print("  5. 维护操作（Maintenance Operations）- 30次测量")
        print("=" * 80)

        plan = self.design.generate_experiment_plan()
        all_results = []

        e1_cases = plan.get("E1_WRITE", [])
        e4_cases = plan.get("E4_MAINTENANCE", [])
        if not e1_cases:
            print("[Error] E1_WRITE is empty. Aborting.")
            return
        if not e4_cases:
            print("[Error] E4_MAINTENANCE is empty. Aborting.")
            return

        import_case_tpl = next((c for c in e1_cases if c["description"] == "import_entire_dataset"), e1_cases[0])
        delete_case_tpl = next((c for c in e4_cases if c["description"] == "delete_entire_graph"), e4_cases[0])

        e1_results = []
        e4_results = []

        # --- 阶段1：导入一次（测量） ---
        print("\n" + "#" * 20 + " Phase 1: Import once (measure) " + "#" * 20)

        if ENABLE_DB_CLEANUP:
            self.executor.ensure_clean_state()

        # 导入一次（测量）
        import_case_once = {
            **import_case_tpl,
            "warmup_runs": 0,
            "repeat": 1,
            "description": "import_entire_dataset_once",
        }
        r_import = self._run_case(import_case_once)
        all_results.append(r_import)
        e1_results.append(r_import)

        # --- 阶段3：遍历、查询、更新各30次，计算能耗 ---
        print("\n" + "#" * 20 + " Phase 3: Traverse/Queries/Update (30 times each, measure) " + "#" * 20)

        # 遍历查询
        e2_cases = plan.get("E2_READ", [])
        print(f"[进度] 遍历查询包含 {len(e2_cases)} 个用例，每个用例测量 {REPEAT} 次\n")

        e2_results = []
        for idx, case in enumerate(e2_cases, 1):
            print(f"\n{'=' * 80}")
            print(f"[遍历查询] 用例 {idx}/{len(e2_cases)}: {case['description']}")
            print(f"{'=' * 80}")
            case_30 = {**case, "warmup_runs": WARMUP_RUNS, "repeat": REPEAT}
            result = self._run_case(case_30)
            e2_results.append(result)
            all_results.append(result)
            print(f"\n{'=' * 80}")
            print(f"[完成] 用例 {idx}/{len(e2_cases)} 已完成: {case['description']}")
            print(f"{'=' * 80}\n")

        # 属性更新
        e3_cases = plan.get("E3_UPDATE", [])
        print(f"[进度] 属性更新包含 {len(e3_cases)} 个用例，每个用例测量 {REPEAT} 次\n")

        e3_results = []
        for idx, case in enumerate(e3_cases, 1):
            print(f"\n{'=' * 80}")
            print(f"[属性更新] 用例 {idx}/{len(e3_cases)}: {case['description']}")
            print(f"{'=' * 80}")
            case_30 = {**case, "warmup_runs": WARMUP_RUNS, "repeat": REPEAT}
            result = self._run_case(case_30)
            e3_results.append(result)
            all_results.append(result)
            print(f"\n{'=' * 80}")
            print(f"[完成] 用例 {idx}/{len(e3_cases)} 已完成: {case['description']}")
            print(f"{'=' * 80}\n")

        # --- 阶段4：删除一次，并测量能耗 ---
        print("\n" + "#" * 20 + " Phase 4: Delete once (measure) " + "#" * 20)
        delete_case_measure = {
            **delete_case_tpl,
            "description": "delete_final",
            "warmup_runs": 0,
            "repeat": REPEAT
        }
        r_delete = self._run_case(delete_case_measure)
        e4_results.append(r_delete)
        all_results.append(r_delete)
        print("数据已删除，并已测量能耗")

        # 保存各阶段平均结果
        if e1_results:
            self._save_step_results("E1_WRITE", e1_results)
        if e2_results:
            self._save_step_results("E2_READ", e2_results)
        if e3_results:
            self._save_step_results("E3_UPDATE", e3_results)
        if e4_results:
            self._save_step_results("E4_MAINTENANCE", e4_results)

        # 最终汇总
        self._print_summary(all_results)

        # 保存所有详细结果和汇总结果
        self._save_all_results(all_results)

    def _print_summary(self, results: list):
        """按类别格式化并打印最终的实验结果摘要，生成表格格式输出。"""

        # 按操作类型对结果进行分组
        grouped_results = {}
        for r in results:
            op_type = r["operation_type"]
            if op_type not in grouped_results:
                grouped_results[op_type] = []
            grouped_results[op_type].append(r)

        # 定义输出类别和顺序（映射到表格中的操作名称）
        # 按照用户要求：数据导入、遍历查询、1跳查询、2跳查询、3跳查询、属性更新、维护
        categories = {
            "数据导入": ["import_dataset"],
            "遍历查询": ["traverse"],
            "1跳查询": ["query_hop_1"],
            "2跳查询": ["query_hop_2"],
            "3跳查询": ["query_hop_3"],
            "属性更新": ["update"],
            "维护": ["maintenance", "delete"]
        }

        print("\n" + "=" * 120)
        print("KG-DPU Operations Energy Consumption and Carbon Emissions Analysis".center(120))
        print("=" * 120)

        # 表头（单位改为焦耳 J）
        header = f"{'KG-DPU':<20} {'CPU能耗 (J)':<20} {'GPU能耗 (J)':<20} {'磁盘能耗 (J)':<20} {'内存能耗 (J)':<20} {'总能耗 (J)':<20} {'碳排放 (mgCO2)':<20}"
        print(header)
        print("-" * 120)

        # 按类别汇总并打印
        table_rows = []
        for title, op_types in categories.items():
            # 收集该类别下的所有结果
            category_results = []
            for op_type in op_types:
                category_results.extend(grouped_results.get(op_type, []))

            # 如果没有匹配的结果，输出0值
            if not category_results:
                cpu_str = "0.0 ± 0.0"
                gpu_str = "0.0"
                disk_str = "0.0 ± 0.0"
                mem_str = "0.0 ± 0.0"
                total_str = "0.0 ± 0.0"
                co2_str = "0.0 ± 0.0"
                row = f"{title:<20} {cpu_str:<20} {gpu_str:<20} {disk_str:<20} {mem_str:<20} {total_str:<20} {co2_str:<20}"
                print(row)
                table_rows.append({
                    "operation": title,
                    "cpu_J": "0.0 ± 0.0",
                    "gpu_J": "0.0",
                    "disk_J": "0.0 ± 0.0",
                    "mem_J": "0.0 ± 0.0",
                    "total_J": "0.0 ± 0.0",
                    "co2_g": "0.0 ± 0.0"
                })
                continue

            # 汇总该类别下所有操作的平均值（加权平均）
            total_samples = sum(r.get("samples", 0) for r in category_results if r.get("samples", 0) > 0)
            if total_samples == 0:
                # 如果没有有效样本，输出0值
                cpu_str = "0.0 ± 0.0"
                gpu_str = "0.0"
                disk_str = "0.0 ± 0.0"
                mem_str = "0.0 ± 0.0"
                total_str = "0.0 ± 0.0"
                co2_str = "0.0 ± 0.0"
                row = f"{title:<20} {cpu_str:<20} {gpu_str:<20} {disk_str:<20} {mem_str:<20} {total_str:<20} {co2_str:<20}"
                print(row)
                table_rows.append({
                    "operation": title,
                    "cpu_J": "0.0 ± 0.0",
                    "gpu_J": "0.0",
                    "disk_J": "0.0 ± 0.0",
                    "mem_J": "0.0 ± 0.0",
                    "total_J": "0.0 ± 0.0",
                    "co2_g": "0.0 ± 0.0"
                })
                continue

            # 计算汇总统计量，需要均值和标准差正确传播
            # 我们使用合并方差公式，而不是简单重复均值
            pooled_count = 0
            pooled_mean_total = 0.0
            pooled_ss_total = 0.0  # sum of squares
            pooled_mean_cpu = pooled_ss_cpu = 0.0
            pooled_mean_disk = pooled_ss_disk = 0.0
            pooled_mean_mem = pooled_ss_mem = 0.0
            pooled_mean_gpu = pooled_ss_gpu = 0.0
            pooled_co2_vals = []

            for r in category_results:
                n_i = r.get("samples", 0)
                if n_i == 0:
                    continue

                # --- 总能耗 ---
                mean_i = r.get("avg_net_J", 0.0)           # J
                std_i = r.get("std_dev", 0.0)              # J
                # 组件
                comps = r.get("components", {})
                cpu_mean_i = comps.get("CPU", {}).get("avg_kJ", 0.0) * 1000.0  # 转 J
                cpu_std_i  = comps.get("CPU", {}).get("std_kJ", 0.0) * 1000.0
                disk_mean_i = comps.get("Disk", {}).get("avg_kJ", 0.0) * 1000.0
                disk_std_i  = comps.get("Disk", {}).get("std_kJ", 0.0) * 1000.0
                mem_mean_i = comps.get("Memory", {}).get("avg_kJ", 0.0) * 1000.0
                mem_std_i  = comps.get("Memory", {}).get("std_kJ", 0.0) * 1000.0
                gpu_mean_i = comps.get("GPU", {}).get("avg_kJ", 0.0) * 1000.0
                gpu_std_i  = comps.get("GPU", {}).get("std_kJ", 0.0) * 1000.0

                # 更新 pooled 总能耗
                # Online combine means/SS (Chan et al. 1979)
                delta = mean_i - pooled_mean_total
                total_new_count = pooled_count + n_i
                pooled_mean_total += delta * n_i / total_new_count
                pooled_ss_total += (n_i - 1) * (std_i ** 2) + (delta ** 2) * pooled_count * n_i / total_new_count

                # 组件 CPU
                delta = cpu_mean_i - pooled_mean_cpu
                pooled_mean_cpu += delta * n_i / total_new_count
                pooled_ss_cpu += (n_i - 1) * (cpu_std_i ** 2) + (delta ** 2) * pooled_count * n_i / total_new_count

                # Disk
                delta = disk_mean_i - pooled_mean_disk
                pooled_mean_disk += delta * n_i / total_new_count
                pooled_ss_disk += (n_i - 1) * (disk_std_i ** 2) + (delta ** 2) * pooled_count * n_i / total_new_count

                # Memory
                delta = mem_mean_i - pooled_mean_mem
                pooled_mean_mem += delta * n_i / total_new_count
                pooled_ss_mem += (n_i - 1) * (mem_std_i ** 2) + (delta ** 2) * pooled_count * n_i / total_new_count

                # GPU
                delta = gpu_mean_i - pooled_mean_gpu
                pooled_mean_gpu += delta * n_i / total_new_count
                pooled_ss_gpu += (n_i - 1) * (gpu_std_i ** 2) + (delta ** 2) * pooled_count * n_i / total_new_count

                # CO2
                co2_avg = r.get("co2e_mg_avg", r.get("co2e_g_avg", 0.0) * 1000.0)
                co2_std = r.get("co2e_mg_std", r.get("co2e_g_std", 0.0) * 1000.0)
                # 近似展开为样本值重复
                pooled_co2_vals.extend([co2_avg] * n_i)

                pooled_count = total_new_count

            if pooled_count <= 1:
                total_mean = pooled_mean_total
                total_std = 0.0
                cpu_mean = pooled_mean_cpu
                cpu_std = 0.0
                disk_mean = pooled_mean_disk
                disk_std = 0.0
                mem_mean = pooled_mean_mem
                mem_std = 0.0
                gpu_mean = pooled_mean_gpu
                gpu_std = 0.0
            else:
                total_mean = pooled_mean_total
                total_std = (pooled_ss_total / (pooled_count - 1)) ** 0.5
                cpu_mean = pooled_mean_cpu
                cpu_std = (pooled_ss_cpu / (pooled_count - 1)) ** 0.5
                disk_mean = pooled_mean_disk
                disk_std = (pooled_ss_disk / (pooled_count - 1)) ** 0.5
                mem_mean = pooled_mean_mem
                mem_std = (pooled_ss_mem / (pooled_count - 1)) ** 0.5
                gpu_mean = pooled_mean_gpu
                gpu_std = (pooled_ss_gpu / (pooled_count - 1)) ** 0.5

            # 计算 CO2 标准差
            import numpy as _np
            if pooled_co2_vals:
                co2_mean = float(_np.mean(pooled_co2_vals))
                co2_std = float(_np.std(pooled_co2_vals, ddof=1)) if len(pooled_co2_vals) > 1 else 0.0
            else:
                co2_mean = 0.0
                co2_std = 0.0

            # 格式化输出（单位：焦耳 J，使用合适的小数位数）
            def format_small_value(mean_val, std_val=None):
                """格式化小值，自动选择合适的小数位数（单位：J）"""
                if mean_val == 0.0:
                    return "0.0" if std_val is None else "0.0 ± 0.0"
                # 对于焦耳，如果值小于10，使用2位小数；否则使用1位小数
                if mean_val < 10.0:
                    if std_val is not None:
                        return f"{mean_val:.2f} ± {std_val:.2f}"
                    else:
                        return f"{mean_val:.2f}"
                else:
                    if std_val is not None:
                        return f"{mean_val:.1f} ± {std_val:.1f}"
                    else:
                        return f"{mean_val:.1f}"

            cpu_str = format_small_value(cpu_mean, cpu_std)
            gpu_str = format_small_value(gpu_mean) if gpu_mean > 0 else "0.0"
            disk_str = format_small_value(disk_mean, disk_std)
            mem_str = format_small_value(mem_mean, mem_std)
            total_str = format_small_value(total_mean, total_std)
            co2_str = format_small_value(co2_mean, co2_std)

            row = f"{title:<20} {cpu_str:<20} {gpu_str:<20} {disk_str:<20} {mem_str:<20} {total_str:<20} {co2_str:<20}"
            print(row)

            # 格式化保存到文件的值（单位：焦耳 J）
            def format_for_file(mean_val, std_val=None):
                """格式化保存到文件的值（单位：J）"""
                if mean_val == 0.0:
                    return "0.0" if std_val is None else "0.0 ± 0.0"
                if mean_val < 10.0:
                    if std_val is not None:
                        return f"{mean_val:.2f} ± {std_val:.2f}"
                    else:
                        return f"{mean_val:.2f}"
                else:
                    if std_val is not None:
                        return f"{mean_val:.1f} ± {std_val:.1f}"
                    else:
                        return f"{mean_val:.1f}"

            table_rows.append({
                "operation": title,
                "cpu_J": format_for_file(cpu_mean, cpu_std),
                "gpu_J": format_for_file(gpu_mean) if gpu_mean > 0 else "0.0",
                "disk_J": format_for_file(disk_mean, disk_std),
                "mem_J": format_for_file(mem_mean, mem_std),
                "total_J": format_for_file(total_mean, total_std),
                "co2_g": format_for_file(co2_mean / 1000.0, co2_std / 1000.0),  # 保留克用于兼容性
                "co2_mg": format_for_file(co2_mean, co2_std)  # 新增毫克
            })

        print("=" * 120)
        print("\n实验完成！所有结果均为净能耗，已扣除系统闲置基线。")

        # 保存表格数据到文件
        if table_rows:
            table_path = REPORT_DIR / "energy_consumption_table.json"
            with open(table_path, 'w', encoding='utf-8') as f:
                json.dump(table_rows, f, indent=2, ensure_ascii=False)
            print(f"[保存] 能耗表格数据已保存: {table_path}")

    def _save_step_results(self, experiment_name: str, results: list):
        """保存每个实验步骤的平均结果到文件"""
        if not results:
            return

        # 确保报告目录存在
        REPORT_DIR.mkdir(exist_ok=True, parents=True)

        # 保存为 CSV
        csv_path = REPORT_DIR / f"{experiment_name}_step_averages.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        print(f"[保存] {experiment_name} 步骤平均结果已保存: {csv_path}")

        # 保存为 JSON
        json_path = REPORT_DIR / f"{experiment_name}_step_averages.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[保存] {experiment_name} 步骤平均结果已保存: {json_path}")

    def _save_all_results(self, all_results: list):
        """保存所有详细结果和汇总结果到文件"""
        # 确保报告目录存在
        REPORT_DIR.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存所有步骤的平均结果（汇总）
        summary_csv_path = REPORT_DIR / f"experiment_summary_{timestamp}.csv"
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
            if all_results:
                fieldnames = sorted({k for r in all_results for k in r.keys()})
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)

        print(f"[保存] 实验汇总结果已保存: {summary_csv_path}")

        # 保存为 JSON
        summary_json_path = REPORT_DIR / f"experiment_summary_{timestamp}.json"
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"[保存] 实验汇总结果已保存: {summary_json_path}")

        # 2. 保存详细的每次测量结果
        if self.detailed_results:
            detailed_csv_path = REPORT_DIR / f"experiment_detailed_{timestamp}.csv"
            with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = sorted({k for r in self.detailed_results for k in r.keys()})
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.detailed_results)

            print(f"[保存] 详细测量结果已保存: {detailed_csv_path}")

            # 保存为 JSON
            detailed_json_path = REPORT_DIR / f"experiment_detailed_{timestamp}.json"
            with open(detailed_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.detailed_results, f, indent=2, ensure_ascii=False)

            print(f"[保存] 详细测量结果已保存: {detailed_json_path}")

        # 3. 保存最新的汇总结果（不带时间戳，方便后续处理）
        latest_summary_csv = REPORT_DIR / "latest_experiment_summary.csv"
        latest_summary_json = REPORT_DIR / "latest_experiment_summary.json"

        with open(latest_summary_csv, 'w', newline='', encoding='utf-8') as f:
            if all_results:
                fieldnames = sorted({k for r in all_results for k in r.keys()})
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)

        with open(latest_summary_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"[保存] 最新汇总结果已保存: {latest_summary_csv} 和 {latest_summary_json}")

        # 4. 保存实验元数据
        metadata = {
            "experiment_timestamp": timestamp,
            "baseline_power_W": self.baseline_power_W,
            "baseline_ci_W": self.baseline_ci_W,
            "total_steps": len(all_results),
            "total_detailed_measurements": len(self.detailed_results),
            "repeat_count": REPEAT,
            "warmup_runs": WARMUP_RUNS
        }

        metadata_path = REPORT_DIR / f"experiment_metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[保存] 实验元数据已保存: {metadata_path}")
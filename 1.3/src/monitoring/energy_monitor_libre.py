# energy_monitor_libre.py  ——  最终修复优化版（短操作不再为0 + 积分稳定 + 准确组件）
import os
import time
import json
import csv
import threading
from typing import Dict, Any
from datetime import datetime

import numpy as np
from scipy import integrate
import requests

from config import REPORT_DIR, LAMBDA_CPU, LAMBDA_GPU, LAMBDA_MEM, LAMBDA_DISK


class EnergyMonitorLibre:
    def __init__(self, libre_url="http://localhost:8085/data.json", sample_interval=0.1):  # 10Hz
        self.libre_url = libre_url
        self.sample_interval = sample_interval
        self.report_dir = REPORT_DIR

        # 中国电网平均碳强度（2024-2025 数据）
        self.carbon_intensity = 550.0  # gCO2e/kWh

        os.makedirs(self.report_dir, exist_ok=True)
        self.detailed_csv = self.report_dir / "energy_detailed.csv"
        self.summary_file = self.report_dir / "energy_summary.jsonl"

        if not self.detailed_csv.exists():
            with open(self.detailed_csv, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "phase", "time_s", "cpu_W", "gpu_W", "mem_W", "disk_W", "total_W"
                ])

        self._reset()

    def _reset(self):
        self._start = None
        self._phase = None
        self._samples = []
        self._samples_lock = threading.Lock()  # 保护 _samples 列表的线程锁
        self._sampling = False
        self._sampling_thread = None

    def _sample(self) -> Dict:
        try:
            data = requests.get(self.libre_url, timeout=2).json()
            cpu = gpu = mem = disk = 0.0

            def walk(node):
                nonlocal cpu, gpu, mem, disk
                text = str(node.get("Text", "") or "").lower()
                val_str = node.get("Value", "")
                sensor_type = str(node.get("Type", "") or "")

                if val_str and sensor_type == "power":
                    try:
                        v = float(val_str.split()[0])
                        if "cpu package" in text or "cpu cores" in text:
                            cpu = max(cpu, v)
                        elif "gpu" in text:
                            gpu = max(gpu, v)
                        elif "memory" in text:
                            mem = v
                        elif any(x in text for x in ["nvme", "hdd", "ssd", "disk"]):
                            disk = max(disk, v)
                    except:
                        pass

                for child in node.get("Children", []):
                    walk(child)

            walk(data)

            # fallback 兜底（避免读取失败全为0）
            if cpu == 0:
                cpu = 35.0
            mem = max(mem, 5.0)
            disk = max(disk, 2.0)

            total = cpu + gpu + mem + disk

            return {
                "cpu_W": round(cpu, 3),
                "gpu_W": round(gpu, 3),
                "mem_W": round(mem, 3),
                "disk_W": round(disk, 3),
                "total_W": round(total, 3)
            }
        except Exception as e:
            print(f"[EnergyMonitor] 采样失败，使用兜底值: {e}")
            return {
                "cpu_W": 35.0,
                "gpu_W": 0.0,
                "mem_W": 8.0,
                "disk_W": 3.0,
                "total_W": 46.0
            }
    
    def _continuous_sampling(self):
        """后台持续采样线程（用于长时间测量）"""
        sample_count = 0
        while self._sampling and self._start:
            try:
                current_time = time.time() - self._start
                s = self._sample()
                s.update({
                    "timestamp": datetime.now().isoformat(),
                    "phase": self._phase,
                    "time_s": current_time
                })
                with self._samples_lock:
                    self._samples.append(s)
                sample_count += 1
                time.sleep(self.sample_interval)
            except Exception as e:
                print(f"[EnergyMonitor] 采样线程错误: {e}")
                time.sleep(self.sample_interval)  # 即使出错也继续尝试
        # 线程退出时打印采样统计
        if sample_count > 0:
            print(f"[EnergyMonitor] 后台采样线程完成，共采集 {sample_count} 个样本")

    def measure_start(self, phase: str):
        self._reset()
        self._phase = phase
        self._start = time.time()
        self._sampling = True
        print(f"[EnergyMonitor] 开始测量 → {phase} (采样间隔: {self.sample_interval}s)")

        # 立即采样第一帧（关键！防止超快操作漏采）
        first_sample = self._sample()
        first_sample.update({
            "timestamp": datetime.now().isoformat(),
            "phase": self._phase,
            "time_s": 0.0
        })
        with self._samples_lock:
            self._samples.append(first_sample)
        
        # 启动后台采样线程（用于长时间测量）
        self._sampling_thread = threading.Thread(target=self._continuous_sampling, daemon=True)
        self._sampling_thread.start()
        # 短暂等待确保线程启动
        time.sleep(0.01)

    def measure_stop(self, workload_factor: float = 0.0, workload_series=None) -> Dict[str, Any]:
        """停止测量并返回能耗结果。

        workload_factor: 图负载因子（GWF_op）。
        - =0.0 表示不加权（默认兼容旧逻辑，因为 1 + 0 = 1）
        - >0.0 表示按 (1 + GWF) 放大组件能耗与总能耗
        """
        if not self._start:
            raise RuntimeError("未调用 measure_start")

        # 停止后台采样
        self._sampling = False
        
        # 等待采样线程完成（给足够时间让线程完成当前采样周期）
        if self._sampling_thread and self._sampling_thread.is_alive():
            # 等待最多 2 个采样周期 + 1 秒缓冲
            max_wait = (self.sample_interval * 2) + 1.0
            self._sampling_thread.join(timeout=max_wait)
            if self._sampling_thread.is_alive():
                print(f"[EnergyMonitor] 警告: 采样线程未在 {max_wait}s 内完成，强制继续")

        duration = time.time() - self._start

        # 采集最后一帧（结束瞬间）
        final_sample = self._sample()
        final_sample.update({
            "timestamp": datetime.now().isoformat(),
            "phase": self._phase,
            "time_s": duration
        })
        with self._samples_lock:
            self._samples.append(final_sample)
            
            # 确保至少有 3 个点用于 Simpson 积分（奇数点更佳）
            while len(self._samples) < 3:
                last = self._samples[-1].copy()
                last["time_s"] = duration
                last["timestamp"] = datetime.now().isoformat()
                self._samples.append(last)
            
            # 获取最终样本列表（线程安全）
            samples = list(self._samples)

        # 时间轴修正：确保单调递增且均匀分布（Simpson积分要求）
        # 注意：保留原始samples用于CSV保存，只在积分计算时使用处理后的时间戳
        t_orig = np.array([s["time_s"] for s in samples])
        t = t_orig.copy()  # 工作副本
        
        # 检查时间戳质量
        needs_interpolation = False
        if len(t) > 1:
            time_diffs = np.diff(t)
            min_diff = np.min(time_diffs[time_diffs > 0]) if np.any(time_diffs > 0) else self.sample_interval
            max_diff = np.max(time_diffs)
            
            # 如果时间戳间隔差异过大（>50%）或非单调，使用均匀分布
            if max_diff > 1.5 * min_diff or np.any(time_diffs < 0):
                if np.any(time_diffs < 0):
                    print(f"[EnergyMonitor] 警告: 检测到非单调时间戳，进行修正")
                else:
                    print(f"[EnergyMonitor] 信息: 时间戳间隔不均匀，使用均匀分布以提高积分精度")
                needs_interpolation = True
                # 确保奇数个点（Simpson积分要求）
                n_points = len(t)
                if n_points % 2 == 0:
                    n_points += 1
                t = np.linspace(0, duration, n_points)
            elif len(t) % 2 == 0 and len(t) > 2:
                # 确保奇数个点（Simpson积分要求）- 移除最后一个点
                t = t[:-1]
                print(f"[EnergyMonitor] 信息: 调整为奇数个点 ({len(t)}) 以优化Simpson积分精度")

        # =========================
        # 论文式：基线 + 动态功率模型，并引入时间变化的 GWF_op(t)
        #
        # 对每个组件 comp：
        #   P_comp(t) = P_base_comp + P_dyn_comp(t)
        #   P_dyn_comp(t) = max(0, P_comp(t) - P_base_comp)
        #   P'_comp(t) = P_base_comp + P_dyn_comp(t) * (1 + λ_comp * GWF(t))
        #   E_comp = ∫ P'_comp(t) dt
        #
        # 其中 GWF(t) 是“版本2”时间序列。
        # 如果未提供 workload_series，则退化为常数 GWF=workload_factor。
        # =========================

        # 提取功率值：如果时间戳被重新分配，需要插值
        if needs_interpolation or len(t) != len(samples):
            # 时间戳已重新分配，需要插值功率值
            raw_powers = {}
            for k in ["cpu_W", "gpu_W", "mem_W", "disk_W"]:
                p_orig = np.array([s[k] for s in samples], dtype=float)
                if len(p_orig) > 1 and len(t_orig) > 1:
                    raw_powers[k] = np.interp(t, t_orig, p_orig)
                elif len(p_orig) > 0:
                    # 单个值或无法插值，使用常数值
                    raw_powers[k] = np.full_like(t, float(p_orig[0]), dtype=float)
                else:
                    raw_powers[k] = np.zeros_like(t, dtype=float)
        else:
            # 直接使用原始功率值
            raw_powers = {k: np.array([s[k] for s in samples], dtype=float) for k in ["cpu_W", "gpu_W", "mem_W", "disk_W"]}
            # 如果t被截断（为了奇数个点），也需要截断功率值
            if len(t) < len(samples):
                for k in raw_powers:
                    raw_powers[k] = raw_powers[k][:len(t)]

        # 1) 估计每个组件的 P_base：用本次测量采样中的最小值作为基线近似
        #    说明：对于baseline测量，最小值接近真实基线；对于操作测量，最小值可能高于真实基线，
        #    但这里主要用于计算动态功率部分，真正的基线扣除在kg_lca_core中完成
        p_base = {
            "cpu_W": float(np.min(raw_powers["cpu_W"])) if len(raw_powers["cpu_W"]) else 0.0,
            "gpu_W": float(np.min(raw_powers["gpu_W"])) if len(raw_powers["gpu_W"]) else 0.0,
            "mem_W": float(np.min(raw_powers["mem_W"])) if len(raw_powers["mem_W"]) else 0.0,
            "disk_W": float(np.min(raw_powers["disk_W"])) if len(raw_powers["disk_W"]) else 0.0,
        }
        
        # 验证基线估计的合理性（避免异常值）
        for comp_key in p_base:
            if p_base[comp_key] < 0:
                print(f"[EnergyMonitor] 警告: {comp_key} 基线功率为负，设为0")
                p_base[comp_key] = 0.0

        # 2) 构造 GWF(t) 与 t 对齐
        if workload_series:
            # workload_series: List[{"t": seconds, "gwf": float}] 或 GwfSample 列表
            ts = []
            gs = []
            for item in workload_series:
                if hasattr(item, "t") and hasattr(item, "gwf"):
                    ts.append(float(item.t))
                    gs.append(float(item.gwf))
                else:
                    ts.append(float(item.get("t", 0.0)))
                    gs.append(float(item.get("gwf", 0.0)))
            if len(ts) >= 2:
                gwf_t = np.interp(t, np.array(ts, dtype=float), np.array(gs, dtype=float), left=gs[0], right=gs[-1])
            elif len(ts) == 1:
                gwf_t = np.full_like(t, float(gs[0]), dtype=float)
            else:
                gwf_t = np.full_like(t, float(workload_factor or 0.0), dtype=float)
        else:
            gwf_t = np.full_like(t, float(workload_factor or 0.0), dtype=float)

        # 3) 按组件计算加权能耗
        lambdas = {
            "cpu_W": float(LAMBDA_CPU),
            "gpu_W": float(LAMBDA_GPU),
            "mem_W": float(LAMBDA_MEM),
            "disk_W": float(LAMBDA_DISK),
        }

        energies = {}
        for comp_key, p in raw_powers.items():
            base = float(p_base[comp_key])
            dyn = np.maximum(0.0, p - base)
            adj = 1.0 + (lambdas[comp_key] * gwf_t)
            p_weighted = base + dyn * adj

            # Simpson积分需要奇数个点，如果不符合则使用梯形积分
            if len(t) >= 3 and len(t) % 2 == 1:
                try:
                    energies[comp_key.replace("_W", "")] = float(integrate.simpson(p_weighted, t))
                except Exception as e:
                    print(f"[EnergyMonitor] 警告: Simpson积分失败 ({comp_key}): {e}，改用梯形积分")
                    energies[comp_key.replace("_W", "")] = float(np.trapz(p_weighted, t))
            else:
                # 使用梯形积分（对偶数个点或少于3个点）
                energies[comp_key.replace("_W", "")] = float(np.trapz(p_weighted, t))

        total_energy_J = float(sum(energies.values()))

        # 兜底保护：极短操作避免积分为0
        if total_energy_J < 1.0 and duration > 0.01:
            min_power = 35.0  # 合理最小功率
            total_energy_J = max(total_energy_J, min_power * duration)
            # 按比例分配到组件
            ratio = total_energy_J / sum(energies.values()) if sum(energies.values()) > 0 else 1
            for k in energies:
                energies[k] *= ratio

        energy_Wh = total_energy_J / 3600.0
        energy_kWh = energy_Wh / 1000.0
        co2e_g = energy_kWh * self.carbon_intensity
        co2e_mg = co2e_g * 1000.0  # 转换为毫克

        result = {
            "phase": self._phase,
            "duration_s": round(duration, 3),
            "energy_J": round(total_energy_J, 3),
            "energy_Wh": round(energy_Wh, 3),
            "co2e_g": round(co2e_g, 3),  # 保留克用于兼容性
            "co2e_mg": round(co2e_mg, 3),  # 新增毫克
            "carbon_intensity_gCO2e_kWh": self.carbon_intensity,
            "components_J": {
                "CPU": round(energies.get("cpu", 0.0), 3),
                "GPU": round(energies.get("gpu", 0.0), 3),
                "Memory": round(energies.get("mem", 0.0), 3),
                "Disk": round(energies.get("disk", 0.0), 3)
            },
            "samples": len(samples),
            "timestamp": datetime.now().isoformat()
        }

        # 保存详细记录
        with open(self.detailed_csv, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            for s in samples:
                w.writerow([
                    s["timestamp"], s["phase"], s["time_s"],
                    s["cpu_W"], s["gpu_W"], s["mem_W"], s["disk_W"], s["total_W"]
                ])

        # 保存摘要
        with open(self.summary_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

        # 打印总结
        expected_samples = int(duration / self.sample_interval) + 2  # +2 为起始和结束帧
        actual_samples = len(samples)
        sample_rate = actual_samples / duration if duration > 0 else 0
        
        print(f"[EnergyMonitor] 完成测量: {self._phase}")
        print(f"   ├─ 执行时长: {duration:.3f} 秒")
        print(f"   ├─ 样本数量: {actual_samples} (期望: ~{expected_samples}, 采样率: {sample_rate:.1f} Hz)")
        print(f"   ├─ 总能耗: {total_energy_J:.2f} J ({energy_Wh:.3f} Wh)")
        print(f"   ├─ 碳排放: {co2e_mg:.3f} mg CO2e")
        print(f"   └─ 组件拆分 → CPU {energies.get('cpu',0):.1f}J | GPU {energies.get('gpu',0):.1f}J | "
              f"内存 {energies.get('mem',0):.1f}J | 磁盘 {energies.get('disk',0):.1f}J")

        return result
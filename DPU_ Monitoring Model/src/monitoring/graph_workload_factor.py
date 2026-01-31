# graph_workload_factor.py
# 计算图负载因子 GWF_op
# GWF_op = α*(D_op/D_max) + β*(E_op/E_total) + γ*(C_op/C_avg)
#
# 说明（和你给的定义一致）：
# - D_op: 本次操作的遍历深度（例如 multi-hop 查询的 hop 数）
# - D_max: 图中理论最大可达路径深度（这里用“图直径”的近似）
# - E_op: 本次操作访问/涉及的边数（用查询返回的 path_count 近似，或用操作类型估计）
# - E_total: 图中总边数
# - C_op: 本次操作导致的聚类系数变化量（这里用 0 作为默认近似；要精确需要更复杂的计算）
# - C_avg: 图的平均聚类系数（这里用 1 兜底，避免除 0）
#
# 注意：Neo4j 上精确计算 D_max、C_avg、C_op 会非常重。
# 为了“简单可控”，这里实现一个可运行的近似版本：
# - D_max: 用 BenchNode 子图上抽样估计（可用 APOC 更准，但本项目不强依赖 APOC）
# - C_avg: 默认 1.0（等你需要论文级聚类系数时再扩展）
# - C_op: 默认 0.0（表示该项暂不参与）

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

from config import ALPHA, BETA, GAMMA


@dataclass
class GraphStats:
    # 图全局统计量
    E_total: int
    D_max: int
    C_avg: float


class GraphWorkloadFactorCalculator:
    def __init__(self, executor):
        self.executor = executor
        self._cache: Optional[GraphStats] = None

    def refresh_graph_stats(self) -> GraphStats:
        # 1) 总边数：只统计测试图相关边（Crop、Disease、Environment）
        # 2) D_max：用一个非常轻量的近似，使用固定值 5 作为默认上限
        # 3) C_avg：使用默认值 1.0 避免复杂计算
        
        # 获取总边数
        try:
            res_edges = self.executor.run_query("count_bench_edges")
            if res_edges and len(res_edges) > 0:
                # Neo4j Record 对象支持字典式访问，使用 data() 转换为字典更安全
                first_record = res_edges[0]
                if hasattr(first_record, 'data'):
                    record_dict = first_record.data()
                    E_total = int(record_dict.get("edges", 0))
                elif isinstance(first_record, dict):
                    E_total = int(first_record.get("edges", 0))
                else:
                    # 尝试直接通过键访问
                    E_total = int(first_record["edges"]) if "edges" in first_record else 0
            else:
                E_total = 0
        except Exception as e:
            print(f"[GWF] 获取边数时出错: {e}，使用默认值 0")
            E_total = 0

        # D_max：使用固定值或查询结果
        try:
            res_d = self.executor.run_query("estimate_bench_diameter")
            if res_d and len(res_d) > 0:
                first_record = res_d[0]
                if hasattr(first_record, 'data'):
                    record_dict = first_record.data()
                    D_max = int(record_dict.get("diameter", 5))
                elif isinstance(first_record, dict):
                    D_max = int(first_record.get("diameter", 5))
                else:
                    D_max = int(first_record["diameter"]) if "diameter" in first_record else 5
            else:
                D_max = 5
        except Exception as e:
            print(f"[GWF] 获取直径时出错: {e}，使用默认值 5")
            D_max = 5

        # C_avg：使用固定值或查询结果
        try:
            res_c = self.executor.run_query("bench_avg_clustering")
            if res_c and len(res_c) > 0:
                first_record = res_c[0]
                if hasattr(first_record, 'data'):
                    record_dict = first_record.data()
                    C_avg = float(record_dict.get("c_avg", 1.0))
                elif isinstance(first_record, dict):
                    C_avg = float(first_record.get("c_avg", 1.0))
                else:
                    C_avg = float(first_record["c_avg"]) if "c_avg" in first_record else 1.0
            else:
                C_avg = 1.0
        except Exception as e:
            print(f"[GWF] 获取聚类系数时出错: {e}，使用默认值 1.0")
            C_avg = 1.0

        self._cache = GraphStats(E_total=E_total, D_max=D_max, C_avg=C_avg)
        return self._cache

    def get_graph_stats(self) -> GraphStats:
        return self._cache or self.refresh_graph_stats()

    def compute_gwf(self, case: Dict[str, Any], op_result: Optional[list] = None) -> float:
        stats = self.get_graph_stats()

        # D_op：从实验用例里拿 hops；如果不是 hop 查询则视作 1
        D_op = int(case.get("hops", 1) or 1)

        # E_op：尽量从查询返回里拿 path_count；拿不到就按操作类型给一个粗略估计
        E_op = 0
        if op_result and len(op_result) > 0:
            first = op_result[0]
            if isinstance(first, dict) and "path_count" in first:
                try:
                    E_op = int(first["path_count"])
                except:
                    E_op = 0

        if E_op <= 0:
            op = case.get("operation")
            size = int(case.get("size", 0) or 0)
            # 很粗的近似：
            if op in ("import_isolated", "import_connected"):
                E_op = max(1, size)  # 创建/连接大致与节点数同阶
            elif op in ("update", "delete"):
                E_op = 1000
            elif op in ("traverse",):
                E_op = max(1, stats.E_total)
            else:  # query_simple / query_agri
                E_op = 100

        # C_op：准确优先 - 需要对比“操作前后”的聚类系数变化。
        # 这里的 compute_gwf() 只在“操作后”拿到 op_result。
        # 所以我们通过在 controller 里传入 pre_stats 来实现（见 kg_lca_core.py 的修改）。
        # 如果未提供，则退化为 0。
        C_op = float(case.get("C_op", 0.0) or 0.0)

        # 防止除 0
        D_max = max(1, stats.D_max)
        E_total = max(1, stats.E_total)
        C_avg = max(1e-9, stats.C_avg)

        gwf = (ALPHA * (D_op / D_max)) + (BETA * (E_op / E_total)) + (GAMMA * (C_op / C_avg))

        # 约束在 [0, 5]：准确优先时允许更大的负载差异，但仍给一个上限防止异常值
        return max(0.0, min(float(gwf), 5.0))


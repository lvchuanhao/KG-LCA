# gwf_sampler.py
# 时间变化的 GWF_op(t) 采样器
#
# 目标：在一次图操作执行期间，按固定频率采样 GWF_op(t)，并与能耗采样对齐。
#
# 约束：Neo4j 的“某条 query 当前跑到哪里/访问了多少边”在数据库层面并不好直接拿到。
# 因此这里采用“可运行且低侵入”的准确优先策略：
# - 使用 Neo4j 内置视图查询当前活动事务（SHOW TRANSACTIONS / dbms.listQueries）
# - 从中读取 elapsedTimeMillis、pageHits/pageFaults、allocatedBytes 等统计（不同版本字段不同）
# - 将这些“运行时负载信号”映射成 E_op(t) 的代理量（proxy），从而形成 GWF_op(t)
# - D_op(t) 可以从 case.hops 视作常数；若是 traverse/query_agri，可设更大权重
# - C_op(t) 难以实时求，准确优先下改为：操作结束后计算一次 ΔC 并把它均匀分摊到时间轴
#
# 这能做到：GWF(t) 随时间变化（由事务运行指标驱动），并且不需要改 Neo4j 内核。

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from monitoring.graph_workload_factor import GraphWorkloadFactorCalculator


@dataclass
class GwfSample:
    t: float  # seconds since start
    gwf: float
    meta: Dict[str, Any]


class GwfTimeSeriesSampler:
    def __init__(self, executor, gwf_calc: GraphWorkloadFactorCalculator, sample_interval: float = 0.2):
        self.executor = executor
        self.gwf_calc = gwf_calc
        self.sample_interval = float(sample_interval)

    def sample_during_operation(
        self,
        case: Dict[str, Any],
        tx_query_id: Optional[str],
        duration_s: float,
        c_op_total: float,
    ) -> List[GwfSample]:
        """在操作执行期间采样 GWF(t)。

        tx_query_id: 目标事务/查询的 id（如果能定位到）；定位不到则采样全局负载代理。
        duration_s: 操作总时长
        c_op_total: 操作完成后计算的总 ΔC（聚类系数变化），会均匀分摊到每个 t
        """
        if duration_s <= 0:
            return [GwfSample(t=0.0, gwf=0.0, meta={"reason": "duration<=0"})]

        # 刷新一次图全局统计（E_total, D_max, C_avg）
        stats = self.gwf_calc.refresh_graph_stats()

        # 均匀分摊 C_op(t)
        c_op_per_t = (c_op_total / max(1, int(duration_s / self.sample_interval))) if c_op_total else 0.0

        samples: List[GwfSample] = []
        start = time.time()

        while True:
            t = time.time() - start
            if t > duration_s:
                break

            # 采样当前运行时负载信号
            runtime = self._read_runtime_signal(tx_query_id)

            # 将 runtime 信号映射成 E_op(t) 的 proxy（准确优先：使用 pageHits/allocatedBytes 等）
            # 数值可能很大，所以做一个平滑归一化。
            e_proxy = float(runtime.get("pageHits", 0) or 0) + 0.001 * float(runtime.get("allocatedBytes", 0) or 0)
            # 防止 0
            e_proxy = max(1.0, e_proxy)

            # 构造一个临时 case（把 E_op 用 proxy 填进去）
            tmp_case = dict(case)
            tmp_case["C_op"] = c_op_per_t
            tmp_case["_E_op_proxy"] = e_proxy

            gwf = self._compute_gwf_with_proxy(tmp_case, stats)

            samples.append(GwfSample(t=float(t), gwf=float(gwf), meta=runtime))
            time.sleep(self.sample_interval)

        if not samples:
            samples.append(GwfSample(t=0.0, gwf=0.0, meta={"reason": "no_samples"}))

        return samples

    def _compute_gwf_with_proxy(self, case: Dict[str, Any], stats) -> float:
        # 复制 graph_workload_factor.compute_gwf 的关键逻辑，但 E_op 用 proxy
        # D_op
        d_op = int(case.get("hops", 1) or 1)

        # E_op
        e_op = float(case.get("_E_op_proxy", 1.0) or 1.0)

        # C_op
        c_op = float(case.get("C_op", 0.0) or 0.0)

        # 防止除 0
        d_max = max(1, int(stats.D_max or 1))
        e_total = max(1, int(stats.E_total or 1))
        c_avg = max(1e-9, float(stats.C_avg or 1.0))

        from config import ALPHA, BETA, GAMMA

        gwf = (ALPHA * (d_op / d_max)) + (BETA * (e_op / e_total)) + (GAMMA * (c_op / c_avg))
        return max(0.0, min(float(gwf), 5.0))

    def _read_runtime_signal(self, tx_query_id: Optional[str]) -> Dict[str, Any]:
        """读取 Neo4j 当前运行时负载信号。

        兼容策略：不同 Neo4j 版本字段不同，尽量抽取常见字段。
        """
        # 优先：SHOW TRANSACTIONS
        try:
            rows = self.executor.run_cypher(
                "SHOW TRANSACTIONS YIELD transactionId, currentQuery, elapsedTimeMillis, pageHits, pageFaults, allocatedBytes "
                "RETURN transactionId, elapsedTimeMillis, pageHits, pageFaults, allocatedBytes"
            )
            # 如果能精确匹配 id，就取那一条；否则取 elapsedTime 最大的那条（通常是正在跑的）
            if rows:
                if tx_query_id:
                    for r in rows:
                        if str(r.get("transactionId")) == str(tx_query_id):
                            return r
                # fallback: longest
                return max(rows, key=lambda x: x.get("elapsedTimeMillis", 0) or 0)
        except Exception:
            pass

        # fallback：老版本 dbms.listQueries
        try:
            rows = self.executor.run_cypher(
                "CALL dbms.listQueries() YIELD queryId, elapsedTimeMillis, pageHits, pageFaults, allocatedBytes "
                "RETURN queryId, elapsedTimeMillis, pageHits, pageFaults, allocatedBytes"
            )
            if rows:
                if tx_query_id:
                    for r in rows:
                        if str(r.get("queryId")) == str(tx_query_id):
                            return r
                return max(rows, key=lambda x: x.get("elapsedTimeMillis", 0) or 0)
        except Exception:
            pass

        return {"elapsedTimeMillis": 0, "pageHits": 0, "pageFaults": 0, "allocatedBytes": 0}


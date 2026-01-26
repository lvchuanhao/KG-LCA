# ------------------------------------------------------------
# node_energy_analyzer.py
# 节点级知识图谱能耗归因分析模块
# ------------------------------------------------------------
import os
import csv
import json
from neo4j import GraphDatabase


class NodeEnergyAnalyzer:
    def __init__(self, uri, user, password, report_dir="kg_lca_reports"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

    def close(self):
        self.driver.close()

    # ----------------------------------------------
    # 抽取节点特征：度数、属性数、标签等
    # ----------------------------------------------
    def get_node_features(self, limit=5000):
        query = """
            MATCH (n)
            RETURN elementId(n) AS id, labels(n) AS labels,
               size([(n)--() | 1]) AS degree,
               size(keys(n)) AS props
            LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, {"limit": limit})
            return [r.data() for r in result]

    # ----------------------------------------------
    # 能耗分配算法
    # energy_total_J: 当前阶段的总能耗
    # node_features: 从 get_node_features() 得到的节点特征
    # ----------------------------------------------
    def distribute_energy(self, energy_total_J, node_features):
        if not node_features:
            return []

        # 用节点复杂度(度数+属性)作为权重
        weights = [f["degree"] + 0.5 * f["props"] + 1 for f in node_features]
        total_weight = sum(weights)

        results = []
        for i, f in enumerate(node_features):
            node_energy = (weights[i] / total_weight) * energy_total_J
            results.append({
                "id": f["id"],
                "labels": ",".join(f["labels"]),
                "degree": f["degree"],
                "props": f["props"],
                "energy_J": node_energy
            })

        results.sort(key=lambda x: x["energy_J"], reverse=True)
        return results

    # ----------------------------------------------
    # 保存节点能耗排名
    # ----------------------------------------------
    def save_rank(self, results, phase):
        out_path = os.path.join(self.report_dir, f"node_energy_rank_{phase}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "labels", "degree", "props", "energy_J"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[EnergyAttribution] ✅ Node energy ranking saved: {out_path}")

        top3 = results[:3]
        print("[EnergyAttribution] 🔍 Top energy-consuming nodes:")
        for i, r in enumerate(top3, 1):
            print(f" {i}. Node#{r['id']} | type={r['labels']} | deg={r['degree']} | props={r['props']} | energy={r['energy_J']:.3f} J")
        return out_path

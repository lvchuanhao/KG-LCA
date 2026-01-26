import math
from neo4j import GraphDatabase

class PerformanceAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ---------------------------
    # 通用执行函数
    # ---------------------------
    def run_query(self, query, params=None):
        with self.driver.session() as session:
            return list(session.run(query, params or {}))

    # ---------------------------
    # 核心图谱特征提取函数
    # ---------------------------
    def get_graph_features(self):
        try:
            features = {}

            # --- 1. 节点与边数量 ---
            nodes = self.run_query("MATCH (n) RETURN count(n) AS c")[0]["c"]
            rels = self.run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
            features["num_nodes"] = nodes
            features["num_rels"] = rels

            # --- 2. 平均度与最大度 ---
            deg_res = self.run_query("""
                MATCH (n)
                WITH n, size([(n)--() | 1]) AS deg
                RETURN avg(coalesce(deg, 0)) AS avg_deg, max(deg) AS max_deg
            """)
            features["avg_degree"] = float(deg_res[0]["avg_deg"] or 0)
            features["max_degree"] = float(deg_res[0]["max_deg"] or 0)

            # --- 3. 平均属性数 ---
            prop_res = self.run_query("""
                MATCH (n)
                WITH n, size(keys(n)) AS prop_count
                RETURN avg(prop_count) AS avg_props
            """)
            features["avg_property_count"] = float(prop_res[0]["avg_props"] or 0)

            # --- 4. Schema 层级深度 ---
            depth_res = self.run_query("""
                MATCH p=(a)-[:BELONGS_TO*]->(b)
                RETURN coalesce(max(length(p)),0) AS depth
            """)
            features["schema_depth"] = int(depth_res[0]["depth"] or 0)

            # --- 5. Triples 密度 ---
            features["triples_per_entity"] = (
                rels / nodes if nodes > 0 else 0
            )

            # --- 6. 删除 GDS 聚类系数部分 ---

            # --- 7. 更新节点比例（需要在执行后传参更新） ---
            features["update_ratio"] = 0.0  # 占位符，控制器会更新

            # --- 8. 基本统计特征补全 ---
            features["join_density"] = rels / (nodes * (nodes - 1)) if nodes > 1 else 0

            # --- 9. Label 熵 ---
            target_labels = ['Action', 'Capability', 'ChipStructure', 'Entity', 'Equipment',
                             'JavaConcept', 'ManufacturingStage', 'Material', 'Method',
                             'Parameter', 'SubMethod', 'Technology']

            label_counts = []
            for label in target_labels:
                try:
                    # 对每个标签单独查询计数
                    count_query = f"MATCH (n:`{label}`) RETURN count(n) AS count"
                    result = self.run_query(count_query)
                    if result and result[0]["count"] > 0:
                        label_counts.append({
                            "label": label,
                            "count": result[0]["count"]
                        })
                except Exception as e:
                    # 如果标签不存在或其他错误，跳过
                    print(f"[PerformanceAnalyzer] ⚠️ Could not count label {label}: {e}")
                    continue

            # 计算熵
            counts = [c["count"] for c in label_counts if c["count"] > 0]
            total = sum(counts)
            entropy = -sum((c / total) * math.log2(c / total) for c in counts) if total > 0 else 0
            features["label_entropy"] = round(entropy, 4)
            features["label_distribution"] = {c["label"]: c["count"] for c in label_counts}

            return features

        except Exception as e:
            print(f"[PerformanceAnalyzer] ⚠️ Graph feature extraction failed: {e}")
            return {}

    # 可由控制器调用以更新 update_ratio
    def update_ratio(self, updated_nodes, total_nodes):
        return updated_nodes / total_nodes if total_nodes > 0 else 0.0

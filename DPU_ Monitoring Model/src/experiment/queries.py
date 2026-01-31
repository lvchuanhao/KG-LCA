"""
Refined Cypher query collection used by the KG-LCA experiment.
Added $reps 参数：重复执行查询 N 次放大工作量。
新增：
- traverse_by_type
- query_hop_1 / 2 / 3 （别名到 hop_*_count）
- 基于 GWF 的统计查询：count_bench_edges、estimate_bench_diameter、bench_avg_clustering
"""

# ==============================================================
# Experiment 2: Read Operations
# ==============================================================

E2_READ_QUERIES = {
    # 全图遍历（节点 & 关系计数）
    "traverse_full": """
        MATCH (n)
        WITH count(n) AS total_nodes
        MATCH ()-[r]->()
        RETURN total_nodes, count(r) AS total_relationships
    """,

    # 按节点标签分类型遍历统计
    "traverse_by_type": """
        MATCH (n)
        WITH labels(n) AS labs
        UNWIND labs AS l
        RETURN l AS label, count(*) AS node_count
    """,

    # ---------- 多跳计数 (重复执行 $reps 次) ----------
    # 参数：$start 起点实体名称  $reps 重复次数  $rels 允许的关系类型列表（可选）

    # 为实验设计中的名称提供别名
    "query_hop_1": """
        WITH range(1, coalesce($reps,1)) AS idx
        UNWIND idx AS _
        MATCH (s:Entity {name: $start_name})-[r]->(t:Entity)
        WHERE $rels IS NULL OR type(r) IN $rels
        RETURN count(t) AS path_count
    """,

"query_hop_2": """
        WITH range(1, coalesce($reps,1)) AS idx
        UNWIND idx AS _
        MATCH (s:Entity {name: $start_name})-[r1]->()-[r2]->(t:Entity)
        WHERE ($rels IS NULL OR type(r1) IN $rels)
          AND ($rels IS NULL OR type(r2) IN $rels)
        RETURN count(t) AS path_count
    """,

"query_hop_3": """
        WITH range(1, coalesce($reps,1)) AS idx
        UNWIND idx AS _
        MATCH (s:Entity {name: $start_name})-[rs*1..3]->(t:Entity)
        WHERE $rels IS NULL OR all(rr IN rs WHERE type(rr) IN $rels)
        RETURN count(t) AS path_count
    """,

    # 图统计查询供 GWF 使用
    "count_bench_edges": """
        MATCH ()-[r]->()
        RETURN count(r) AS edges
    """,
    "estimate_bench_diameter": """
        // 简化：返回固定近似 5
        RETURN 5 AS diameter
    """,
    "bench_avg_clustering": """
        // 简化：返回 1.0 作为平均聚类系数
        RETURN 1.0 AS c_avg
    """,
}

# ==============================================================
# Experiment 3: Update Operations
# ==============================================================

E3_UPDATE_QUERIES = {
    "update_nodes": """
        MATCH (n:Entity)
        SET n.hit = coalesce(n.hit, 0) + 1, n.updated_at = datetime()
        RETURN count(n) AS updated_nodes
    """,
    "update_relationships": """
        MATCH ()-[r]->()
        SET r.weight = coalesce(r.weight, 1.0) * 1.1, r.updated_at = datetime()
        RETURN count(r) AS updated_relationships
    """,
}

# ==============================================================
# Experiment 4: Maintenance Operations
# ==============================================================

E4_MAINTENANCE_QUERIES = {
    "delete_all": """
        MATCH (n)
        WITH n LIMIT 10000
        DETACH DELETE n
        RETURN count(n) AS deleted_nodes
    """,
}

# ==============================================================
# Final export dictionary
# ==============================================================

ALL_QUERIES = {}
ALL_QUERIES.update(E2_READ_QUERIES)
ALL_QUERIES.update(E3_UPDATE_QUERIES)
ALL_QUERIES.update(E4_MAINTENANCE_QUERIES)

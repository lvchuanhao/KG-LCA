import re
from pathlib import Path
from typing import Dict, Any, Iterable, List

import pandas as pd


def normalize_relation(rel: str) -> str:
    # 移除所有单引号
    rel = rel.replace("'", "")
    # 只保留字母、数字、空格和下划线，将其他字符替换为下划线
    rel = re.sub(r"[^\w\s]", "_", rel)
    # 转换为大写并将空格替换为下划线
    rel = rel.strip().upper().replace(" ", "_")
    # 移除连续的下划线
    rel = re.sub(r"_+", "_", rel)
    # 如果以数字开头，添加前缀
    if rel and rel[0].isdigit():
        rel = f"REL_{rel}"
    return rel


def _chunks(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def import_entire_dataset_csv(driver, csv_path: str, batch_size: int = 10000) -> Dict[str, Any]:
    """
    只支持“整套数据集导入”：
    - CSV 列：entity1, entity2, relation
    - 创建节点：(:Entity {name})（使用 MERGE，查重）
    - 创建关系：(a)-[:<REL_TYPE>]->(b)   ← 使用动态关系类型（使用 MERGE，查重）
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"数据集文件不存在: {csv_path}")

    df = pd.read_csv(csv_file, sep=",", quotechar='"', on_bad_lines="skip", dtype=str)
    required = {"entity1", "entity2", "relation"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV 需要包含列: {sorted(required)}，实际列: {list(df.columns)}")

    def safe_str(v) -> str:
        if pd.isna(v) or v is None:
            return ""
        return str(v).strip()

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        e1 = safe_str(row.get("entity1"))
        e2 = safe_str(row.get("entity2"))
        rel_raw = safe_str(row.get("relation"))
        if not e1 or not e2 or not rel_raw:
            continue
        rel = normalize_relation(rel_raw)
        if not rel:
            continue
        rows.append({"e1": e1, "e2": e2, "rel": rel})

    # 只取前 N 条有效行（用户要求：一次实验导入 10000 条）
    from config import DATASET_IMPORT_LIMIT
    if DATASET_IMPORT_LIMIT and DATASET_IMPORT_LIMIT > 0:
        rows = rows[: DATASET_IMPORT_LIMIT]

    total_rels = 0
    total_rows = len(rows)
    if batch_size is None or batch_size <= 0:
        batch_size = 10000
    total_batches = (total_rows + batch_size - 1) // batch_size if total_rows > 0 else 0

    print(f"[导入] 有效行数: {total_rows}, 批大小: {batch_size}, 预计批次数: {total_batches}")

    imported_rows = 0

    with driver.session() as session:
        for batch_idx, batch in enumerate(_chunks(rows, batch_size=batch_size), start=1):
            imported_rows += len(batch)

            # 按关系类型分组：不依赖 APOC 的前提下，动态关系类型必须写死在 Cypher 里
            # 这里每个关系类型在一个 batch 内只发送一次 UNWIND，减少网络往返
            rel_groups: Dict[str, List[Dict[str, Any]]] = {}
            for item in batch:
                rel_groups.setdefault(item["rel"], []).append(item)

            created_in_batch = 0
            for rel, items in rel_groups.items():
                cypher = f"""
                UNWIND $batch AS row
                MERGE (a:Entity {{name: row.e1}})
                MERGE (b:Entity {{name: row.e2}})
                MERGE (a)-[r:{rel}]->(b)
                RETURN count(r) AS created
                """
                result = session.run(cypher, batch=[{"e1": it["e1"], "e2": it["e2"]} for it in items])
                rec = result.single()
                created = int(rec["created"]) if rec and rec.get("created") is not None else 0
                created_in_batch += created

            total_rels += created_in_batch

            progress = (imported_rows / total_rows * 100) if total_rows else 100.0
            print(
                f"[进度] 批次 {batch_idx}/{total_batches}，已处理 {imported_rows}/{total_rows} 行 "
                f"({progress:.1f}%)，累计关系: {total_rels}",
                flush=True,
            )

    print(f"[导入完成] 总关系数: {total_rels}，总批次数: {total_batches}")

    return {
        "csv_path": str(csv_file),
        "rows_read": int(len(df)),
        "rows_valid": int(len(rows)),
        "created_relationships": int(total_rels),
        "batches": int(total_batches),
        "batch_size": int(batch_size),
    }

# 兼容旧代码：提供 import_triples_csv 别名（参数一致）

def import_triples_csv(
    driver,
    csv_path: str,
    batch_size: int = 10000,
    limit: int | None = None,
):
    # limit 参数仅作占位，导入函数内部已经在 settings 中控制行数，可忽略
    return import_entire_dataset_csv(driver, csv_path=csv_path, batch_size=batch_size)
"""
仅保留：整套数据集导入（其他导入方式全部移除）。

数据源：`config.settings.DATASET_CSV_PATH`（默认 `data/AgriKG.csv`，列：entity1, entity2, relation）
导入目标：Neo4j 中的 (:Entity {name}) 以及动态关系类型
"""

from neo4j import GraphDatabase

from config import DATASET_CSV_PATH, DATASET_IMPORT_BATCH_SIZE, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.core.dataset_importer import import_entire_dataset_csv

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        stats = import_entire_dataset_csv(driver, csv_path=DATASET_CSV_PATH, batch_size=DATASET_IMPORT_BATCH_SIZE)
        print("✅ 整套数据集导入完成：")
        for k, v in stats.items():
            print(f"  - {k}: {v}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
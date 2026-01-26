# experimental_design.py
# 实验设计：只包含核心操作类型
from typing import List, Dict, Any

class ExperimentalDesign:
    def generate_experiment_plan(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        生成结构化的实验计划，包含以下操作类型：
        1. 数据导入（Data Import）
        2. 遍历查询（Traverse Queries）
        3. 多跳查询（1-hop, 2-hop, 3-hop Queries）
        4. 属性更新（Property Updates）
        5. 维护操作（Maintenance Operations）
        """
        plan = {
            # 实验一：数据导入操作
            "E1_WRITE": [
                # 只保留：整套数据集导入（其他导入全部禁用）
                {"description": "import_entire_dataset", "query": "import_dataset", "params": {}}
            ],
            
            # 实验二：读取操作（遍历查询 + 多跳查询）
            "E2_READ": [
                # 遍历查询
                {"description": "traverse_full_graph", "query": "traverse_full", "params": {}},
                {"description": "traverse_by_type", "query": "traverse_by_type", "params": {}},
                # 多跳查询（使用农业知识图谱中的高频实体作为起点）
                # 根据数据集分析，"菊糖"、"草药"、"茄子"等实体出度较高，适合作为查询起点
                {"description": "1-hop_query", "query": "query_hop_1", "params": {"start_name": "菊糖", "reps": 1, "rels": None}},
                {"description": "2-hop_query", "query": "query_hop_2", "params": {"start_name": "菊糖", "reps": 1, "rels": None}},
                {"description": "3-hop_query", "query": "query_hop_3", "params": {"start_name": "菊糖", "reps": 1, "rels": None}},
            ],
            
            # 实验三：属性更新操作
            "E3_UPDATE": [
                # 只保留：实体属性更新（其他更新全部禁用）
                {"description": "update_entity_properties", "query": "update_nodes", "params": {}}, 
            ],
            
            # 实验四：维护操作
            "E4_MAINTENANCE": [
                # 只保留：删除整张图（其他维护全部禁用）
                {"description": "delete_entire_graph", "query": "delete_all", "params": {}},
            ]
        }
        
        # 确保所有用例都有 params 字段
        for group in plan.values():
            for case in group:
                if "params" not in case:
                    case["params"] = {}
        
        return plan

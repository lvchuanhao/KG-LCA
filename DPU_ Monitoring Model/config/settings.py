# config.py
from pathlib import Path

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "20050924l"

REPEAT = 30  # 30次重复，CI 极小
WARMUP_RUNS = 15
RANDOM_SEED = 42

ISOLATED_SIZES = [100, 1000, 5000]
CONNECTED_SIZE = 1000

REPORT_DIR = Path("kg_lca_reports")
REPORT_DIR.mkdir(exist_ok=True, parents=True)

CONFIDENCE_LEVEL = 0.95

# ==========================
# 数据集导入配置
# ==========================
# 只允许“整套数据集导入”，其他导入方式全部禁用。
# 默认使用仓库内的 data/AgriKG.csv（列：entity1, entity2, relation）
# 使用项目根目录的绝对路径，避免工作目录不同导致找不到文件
DATASET_CSV_PATH = str((Path(__file__).resolve().parent.parent / "data" / "AgriKG.csv"))

# 导入批量大小（每个事务处理多少行）
DATASET_IMPORT_BATCH_SIZE = 10000

# 每次实验实际导入的行数上限：0 表示不限制（导入全部数据）
DATASET_IMPORT_LIMIT = 0

# =========================
# Graph Workload Factor (GWF) 权重系数
# GWF_op = α*(D_op/D_max) + β*(E_op/E_total) + γ*(C_op/C_avg)
# 默认给一个均匀权重；你也可以通过实验校准它们
ALPHA = 1/3
BETA = 1/3
GAMMA = 1/3

# ==========================
# 动态功率模型 (P_dyn) 加权系数 λ
# P'_comp(t) = P_base_comp + P_dyn_comp(t) * (1 + λ_comp * GWF_op(t))
# 默认都设为 1.0，你可以根据实验校准
# ==========================
LAMBDA_CPU = 1.0
LAMBDA_GPU = 1.0
LAMBDA_MEM = 1.0
LAMBDA_DISK = 1.0

# GWF_op(t) 采样器的时间间隔（秒）
GWF_SAMPLE_INTERVAL = 0.2

# ==========================
# 数据库清理控制
# ==========================
# 是否在执行操作前清理数据库
# True: 每次操作前清理（确保干净状态，但会消耗时间）
# False: 不清理（更快，但可能受之前数据影响）
ENABLE_DB_CLEANUP = False  # 设置为 True 启用清理操作

# ==========================
# 跳过整套数据导入控制
# True: run_scientific_experiment 将跳过导入阶段
# False: 正常导入一次并测量能耗
SKIP_IMPORT = True
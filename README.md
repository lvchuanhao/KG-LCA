好，下面给你的是**完整单文件中文 README.md**，已经整理成**一次复制即可直接保存成文件的版本**，**不中断、不拆分**。

👉 操作方式：
1️⃣ 新建文件：`README.md`
2️⃣ 从下面代码块**全部复制**
3️⃣ 粘贴保存
4️⃣ 直接上传 GitHub 即可

---

````md
# KG-LCA —— 知识图谱能耗与碳排放评估框架

KG-LCA 是一个面向**知识图谱操作生命周期（Lifecycle Operations）**的能耗与碳排放测量框架，适用于 Neo4j 等图数据库环境。  
该系统支持对数据导入、图遍历、多跳查询、属性更新和维护操作进行**精细化能源监控与碳足迹核算**。

本项目主要用于**绿色计算研究、图数据库能效评估以及跨硬件平台性能与碳排放对比实验**。

---

## 功能特性

- 知识图谱操作能耗自动测量  
- CPU / 内存 / 磁盘多维度能耗拆分  
- 空闲基线功率校准（净能耗模型）  
- 多轮重复实验统计分析  
- 碳排放自动换算（CO₂ 当量）  
- 支持多硬件平台对比实验  
- 非侵入式监控（无需修改数据库内核）  

---

## 支持的操作类型

| 操作类型 | 支持情况 |
---------|---------
数据导入 | ✓  
图遍历查询 | ✓  
1-hop 查询 | ✓  
2-hop 查询 | ✓  
3-hop 查询 | ✓  
属性更新 | ✓  
图清理/删除 | ✓  

---

## 运行环境要求

### 软件环境

- Python >= 3.9  
- Neo4j 4.x / 5.x  

---

### 推荐环境配置（Conda）

```bash
conda create -n kg-lca python=3.10
conda activate kg-lca
````

---

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖库包括：

* numpy
* scipy
* psutil
* neo4j
* pandas

---

## 系统配置

### 数据库连接配置

编辑 `config.py`：

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
```

---

### 实验参数配置

示例：

```python
REPEAT = 30
WARMUP_RUNS = 5
ENABLE_DB_CLEANUP = True
```

参数说明：

| 参数名               | 说明          |
| ----------------- | ----------- |
| REPEAT            | 每个操作的重复测量次数 |
| WARMUP_RUNS       | 正式测量前的预热次数  |
| ENABLE_DB_CLEANUP | 是否自动清空数据库   |

---

## 运行方法

在项目根目录执行：

```bash
python main.py
```

系统将自动完成：

* 空闲基线功率测量
* 知识图谱工作负载执行
* 硬件资源能耗采样
* 净能耗计算
* 碳排放换算
* 多轮统计分析
* 实验结果保存

---

## 输出结果说明

所有实验结果保存在：

```
report/
```

### 主要输出文件

| 文件名                           | 说明        |
| ----------------------------- | --------- |
| latest_experiment_summary.csv | 实验汇总结果    |
| experiment_detailed_*.csv     | 单次实验原始数据  |
| experiment_metadata.json      | 实验配置与环境信息 |

---

## 碳排放计算模型

系统默认碳强度参数：

```
550 gCO2 / kWh
```

换算公式：

```
CO2 = 能耗(J) / 3.6e6 × 碳强度
```

可在以下文件中修改：

```
core/kg_lca_core.py
```

示例：

```python
carbon_intensity = 550.0
```

---

## 项目结构说明

```
project/
├── main.py
├── config.py
├── core/
│   ├── kg_lca_core.py
│   └── cypher_executor.py
├── monitoring/
│   └── energy_monitor_libre.py
├── experiment/
│   ├── experimental_design.py
│   └── baseline_comparison.py
└── report/
```

---

## 典型应用场景

* 知识图谱系统能效评估
* 绿色计算与绿色数据库研究
* 不同硬件平台能耗对比
* 碳足迹建模与分析
* 学术实验复现
* 系统性能与节能优化

---

## 实验建议

为了保证实验结果稳定性与可复现性：

* 实验期间关闭无关后台程序
* 固定 CPU 频率调节策略
* 多平台使用相同数据集
* 避免设备过热降频
* 建议进行多轮重复实验

---

## 许可说明

本项目仅限**科研与教学用途**使用。
如需商业用途，请联系作者授权。

---

## 学术引用

如果在论文或科研工作中使用本框架，请引用：

```
KG-LCA: 知识图谱操作能耗与碳排放评估框架
```

---

如果本项目对你的研究有所帮助，欢迎 Star 支持。

```

---

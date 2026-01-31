# KG-LCA — Energy Consumption and Carbon Emission Assessment Framework for Knowledge Graphs

**KG-LCA** is a framework for measuring **energy consumption and carbon emissions across the lifecycle operations of knowledge graphs**, designed for graph database environments such as **Neo4j**.

The framework supports **fine-grained energy monitoring and carbon footprint accounting** for key knowledge graph operations, including data import, graph traversal, multi-hop queries, property updates, and maintenance tasks.

This project is primarily intended for **green computing research, graph database energy efficiency evaluation, and cross-hardware platform performance and carbon emission comparison experiments**.

---

## Features

- Automated energy measurement for knowledge graph operations  
- Multi-dimensional energy breakdown (CPU / memory / disk)  
- Idle baseline power calibration (net energy consumption model)  
- Multi-round repeated experiment statistical analysis  
- Automatic carbon emission conversion (CO₂ equivalent)  
- Support for cross-hardware platform comparison experiments  
- Non-intrusive monitoring (no modification to database kernel required)  

---

## Supported Operation Types

| Operation Type | Supported |
|---------------|-----------|
| Data import | ✓ |
| Graph traversal | ✓ |
| 1-hop query | ✓ |
| 2-hop query | ✓ |
| 3-hop query | ✓ |
| Property update | ✓ |
| Graph cleanup / delete | ✓ |

---

## Runtime Environment Requirements

### Software Environment

- Python ≥ 3.9  
- Neo4j 4.x / 5.x  

---

### Recommended Conda Environment

```bash
conda create -n kg-lca python=3.10
conda activate kg-lca
````

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

* numpy
* scipy
* psutil
* neo4j
* pandas

---

## System Configuration

### Database Connection Configuration

Edit `config.py`:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
```

---

### Experiment Parameter Configuration

Example:

```python
REPEAT = 30
WARMUP_RUNS = 5
ENABLE_DB_CLEANUP = True
```

Parameter descriptions:

| Parameter Name    | Description                                      |
| ----------------- | ------------------------------------------------ |
| REPEAT            | Number of repeated measurements per operation    |
| WARMUP_RUNS       | Number of warm-up runs before formal measurement |
| ENABLE_DB_CLEANUP | Whether to automatically clean the database      |

---

## How to Run

Execute the following command in the project root directory:

```bash
python main.py
```

The system will automatically perform:

* Idle baseline power measurement
* Knowledge graph workload execution
* Hardware resource energy sampling
* Net energy consumption calculation
* Carbon emission conversion
* Multi-round statistical analysis
* Experimental result persistence

---

## Output Description

All experimental results are saved in:

```
report/
```

### Main Output Files

| File Name                     | Description                                       |
| ----------------------------- | ------------------------------------------------- |
| latest_experiment_summary.csv | Summary of experimental results                   |
| experiment_detailed_*.csv     | Raw data of individual experiments                |
| experiment_metadata.json      | Experiment configuration and environment metadata |

---

## Carbon Emission Calculation Model

Default carbon intensity parameter:

```
550 gCO2 / kWh
```

Conversion formula:

```
CO2 = Energy (J) / 3.6e6 × Carbon Intensity
```

You can modify the carbon intensity parameter in:

```
core/kg_lca_core.py
```

Example:

```python
carbon_intensity = 550.0
```

---

## Project Structure

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

## Typical Application Scenarios

* Energy efficiency evaluation of knowledge graph systems
* Green computing and green database research
* Cross-hardware platform energy consumption comparison
* Carbon footprint modeling and analysis
* Reproducible academic experiments
* System performance and energy optimization

---

## Experimental Recommendations

To ensure stability and reproducibility of experimental results:

* Close unnecessary background processes during experiments
* Fix CPU frequency scaling policies
* Use the same dataset across different platforms
* Avoid thermal throttling due to overheating
* Perform multiple rounds of repeated experiments

---

## License

This project is intended **for research and educational purposes only**.
For commercial usage, please contact the author for authorization.

---

## Citation

If you use this framework in academic publications or research work, please cite:

```
KG-LCA: An Energy and Carbon Emission Assessment Framework for Knowledge Graph Operations
```

---

If this project is helpful to your research, feel free to ⭐ star the repository.

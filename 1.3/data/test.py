import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv


def calculate_graph_metrics(csv_file):
    """
    计算知识图谱的指标：实体数、关系数、度数分布、图密度等

    参数:
    csv_file: CSV文件路径
    """
    # 1. 读取CSV文件 - 使用更灵活的方式
    try:
        # 方法1: 使用Python标准csv库读取，查看数据结构
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = list(reader)

        print(f"文件总行数: {len(lines)}")
        print("前5行数据:")
        for i in range(min(5, len(lines))):
            print(f"行{i + 1}: {lines[i]}, 字段数: {len(lines[i])}")

        # 查找有问题的行
        problematic_lines = []
        for i, line in enumerate(lines):
            if len(line) != 3:
                problematic_lines.append((i + 1, line))

        if problematic_lines:
            print(f"\n发现 {len(problematic_lines)} 行数据字段数不为3:")
            for line_num, line_content in problematic_lines[:5]:  # 只显示前5个问题
                print(f"  第{line_num}行: {line_content}")

            # 尝试修复：只取前3个字段
            print("\n尝试修复数据...")
            cleaned_lines = []
            for line in lines:
                if len(line) >= 3:
                    cleaned_lines.append(line[:3])  # 只取前3个字段
                else:
                    print(f"跳过第{lines.index(line) + 1}行: 字段不足3个")

        # 创建DataFrame
        df = pd.DataFrame(cleaned_lines, columns=['entity1', 'relation', 'entity2'])

    except Exception as e:
        print(f"读取文件失败: {e}")
        print("\n尝试使用不同编码读取...")
        # 尝试不同编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding, header=None,
                                 names=['entity1', 'relation', 'entity2'],
                                 on_bad_lines='skip')  # 跳过有问题的行
                print(f"使用 {encoding} 编码成功读取")
                break
            except:
                continue
        else:
            print("所有编码尝试失败")
            return None, None

    print(f"\n成功读取数据，共 {len(df)} 条三元组")
    print("数据示例：")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"实体1列唯一值数量: {df['entity1'].nunique()}")
    print(f"关系列唯一值数量: {df['relation'].nunique()}")
    print(f"实体2列唯一值数量: {df['entity2'].nunique()}")

    # 2. 检查数据质量
    print("\n" + "=" * 50)
    print("数据质量检查")
    print("=" * 50)

    # 检查空值
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"发现空值:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col}: {count} 个空值")
        # 删除空值
        df = df.dropna()
        print(f"删除空值后剩余 {len(df)} 条数据")

    # 检查重复数据
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"发现 {duplicates} 条重复数据")
        df = df.drop_duplicates()
        print(f"删除重复后剩余 {len(df)} 条数据")

    # 3. 构建有向图
    G = nx.DiGraph()
    print(f"\n正在构建图...")

    added_edges = 0
    for _, row in df.iterrows():
        G.add_edge(row['entity1'], row['entity2'], relation=row['relation'])
        added_edges += 1

    print(f"图构建完成，添加了 {added_edges} 条边")

    # 4. 计算基本指标
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print("\n" + "=" * 50)
    print("知识图谱指标分析")
    print("=" * 50)
    print(f"1. 实体数 (Entities): {num_nodes}")
    print(f"2. 关系数 (Relations): {num_edges}")

    if num_nodes > 0:
        print(f"3. 平均关系数/实体: {num_edges / num_nodes:.2f}")
    else:
        print(f"3. 平均关系数/实体: N/A (无实体)")

    # 5. 度数分析
    if num_nodes > 0:
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        in_degrees = [G.in_degree(n) for n in G.nodes()]

        print("\n4. 度数分布分析:")
        print(f"   平均出度: {np.mean(out_degrees):.2f} (±{np.std(out_degrees):.2f})")
        print(f"   平均入度: {np.mean(in_degrees):.2f} (±{np.std(in_degrees):.2f})")
        print(f"   最大出度: {max(out_degrees)}")
        print(f"   最大入度: {max(in_degrees)}")
        print(f"   出度为0的节点数: {sum(1 for d in out_degrees if d == 0)}")
        print(f"   入度为0的节点数: {sum(1 for d in in_degrees if d == 0)}")

        # 6. 度数分布直方图
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        max_out_degree = max(out_degrees)
        bins_out = min(20, max_out_degree + 1) if max_out_degree > 0 else 10
        plt.hist(out_degrees, bins=bins_out, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Out-degree Distribution')
        plt.xlabel('Out-degree')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        max_in_degree = max(in_degrees)
        bins_in = min(20, max_in_degree + 1) if max_in_degree > 0 else 10
        plt.hist(in_degrees, bins=bins_in, alpha=0.7, color='red', edgecolor='black')
        plt.title('In-degree Distribution')
        plt.xlabel('In-degree')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('degree_distribution.png', dpi=150)
        print(f"   度数分布图已保存为: degree_distribution.png")

        # 7. 图密度
        if num_nodes > 1:
            # 有向图密度
            max_possible_edges = num_nodes * (num_nodes - 1)
            directed_density = num_edges / max_possible_edges

            # 无向图密度
            G_undirected = G.to_undirected()
            undirected_density = nx.density(G_undirected)

            print(f"\n5. 图密度分析:")
            print(f"   有向图密度: {directed_density:.6f}")
            print(f"   无向图密度: {undirected_density:.6f}")
        else:
            print("\n5. 图密度分析: 节点数太少，无法计算密度")
            directed_density = 0
            undirected_density = 0

        # 8. 连通性分析
        print(f"\n6. 连通性分析:")
        print(f"   弱连通分量数: {nx.number_weakly_connected_components(G)}")
        print(f"   强连通分量数: {nx.number_strongly_connected_components(G)}")

        # 获取最大连通分量
        if nx.is_weakly_connected(G):
            print(f"   图是弱连通的")
        else:
            # 找到最大弱连通分量
            wcc_sizes = [len(cc) for cc in nx.weakly_connected_components(G)]
            if wcc_sizes:
                print(f"   最大弱连通分量大小: {max(wcc_sizes)} 个节点")
                print(f"   连通分量大小分布: {sorted(wcc_sizes, reverse=True)[:5]}...")

        # 9. 关系类型分析
        relation_counts = df['relation'].value_counts()
        print(f"\n7. 关系类型分析:")
        print(f"   关系类型总数: {len(relation_counts)}")
        print(f"   最常用的关系类型 (前10):")
        for i, (rel, count) in enumerate(relation_counts.head(10).items()):
            percentage = count / len(df) * 100
            print(f"     {i + 1}. {rel}: {count} 次 ({percentage:.1f}%)")

        # 10. 保存详细结果
        with open('graph_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("知识图谱分析报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"数据文件: {csv_file}\n")
            f.write(f"分析时间: {pd.Timestamp.now()}\n")
            f.write(f"总行数: {len(lines)}\n")
            f.write(f"有效三元组数: {len(df)}\n\n")

            f.write("基本统计:\n")
            f.write(f"- 实体数: {num_nodes}\n")
            f.write(f"- 关系数: {num_edges}\n")
            if num_nodes > 0:
                f.write(f"- 平均关系数/实体: {num_edges / num_nodes:.2f}\n\n")

            f.write("度数分布:\n")
            f.write(f"- 平均出度: {np.mean(out_degrees):.2f}\n")
            f.write(f"- 平均入度: {np.mean(in_degrees):.2f}\n")
            f.write(f"- 最大出度: {max(out_degrees)}\n")
            f.write(f"- 最大入度: {max(in_degrees)}\n\n")

            f.write("度数分布详情:\n")
            f.write("出度分布:\n")
            degree_freq = {}
            for degree in out_degrees:
                degree_freq[degree] = degree_freq.get(degree, 0) + 1

            for degree in sorted(degree_freq.keys()):
                f.write(f"  出度={degree}: {degree_freq[degree]} 个节点\n")

            f.write("\n入度分布:\n")
            degree_freq_in = {}
            for degree in in_degrees:
                degree_freq_in[degree] = degree_freq_in.get(degree, 0) + 1

            for degree in sorted(degree_freq_in.keys()):
                f.write(f"  入度={degree}: {degree_freq_in[degree]} 个节点\n")

            f.write("\n关系类型统计:\n")
            for rel, count in relation_counts.items():
                percentage = count / len(df) * 100
                f.write(f"  {rel}: {count} 次 ({percentage:.2f}%)\n")

        print(f"\n详细分析报告已保存为: graph_analysis_report.txt")

        # 11. 返回所有计算结果
        results = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_out_degree': np.mean(out_degrees) if num_nodes > 0 else 0,
            'avg_in_degree': np.mean(in_degrees) if num_nodes > 0 else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'directed_density': directed_density,
            'undirected_density': undirected_density,
            'relation_types': len(relation_counts)
        }

        return results, G

    else:
        print("图中没有节点，无法进行分析")
        return None, None


# 使用示例
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    csv_file_path = "AgriKG.csv"  # 请替换为实际文件路径

    # 计算指标
    print(f"开始分析文件: {csv_file_path}")
    results, graph = calculate_graph_metrics(csv_file_path)

    if results:
        print("\n" + "=" * 50)
        print("主要指标摘要:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        print("分析失败")
# cypher_executor.py
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from config import DATASET_CSV_PATH, DATASET_IMPORT_BATCH_SIZE
from experiment.queries import ALL_QUERIES
from core.dataset_importer import import_triples_csv as import_entire_dataset_csv

class CypherExecutor:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def run_query(self, name: str, params=None, max_retries=3, retry_delay=2.0):
        """
        执行查询，带重试机制处理数据库不可用错误
        
        Args:
            name: 查询名称
            params: 查询参数
            max_retries: 最大重试次数（默认3次）
            retry_delay: 重试延迟（秒，默认2秒）
        """
        # 特殊处理：整套数据集导入（由 Python 侧执行，导入前先清空数据库）
        if name == "import_dataset":
            # 先清空数据库，确保导入时处于空状态
            self.ensure_clean_state()
            csv_path = (params or {}).get("csv_path") or DATASET_CSV_PATH
            batch_size = int((params or {}).get("batch_size") or DATASET_IMPORT_BATCH_SIZE)
            result = import_entire_dataset_csv(self.driver, csv_path=csv_path, batch_size=batch_size)
            # 保持与 run_query 其他返回类型一致：list[record-like]
            return [result]

        cypher = ALL_QUERIES.get(name)
        if not cypher:
            raise KeyError(f"Query '{name}' not found!")

        # 特殊处理：delete_graph 需要循环执行直到删除完成
        if name in ["delete_graph"]:
            return self._run_batch_delete_query(name, cypher, params)

        # 处理多语句查询（用分号分隔）
        statements = [s.strip() for s in cypher.strip().split(';') if s.strip()]
        
        # 对于可能较慢的查询，添加时间监控（简化：移除进度提示以提升速度）
        import time
        slow_queries = []  # 已优化查询，不再需要特殊提示
        is_slow_query = False  # 禁用慢查询检测以提升速度
        
        # 重试机制
        last_exception = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                with self.driver.session() as session:
                    # 如果只有一个语句，直接执行
                    if len(statements) == 1:
                        # 添加进度提示（每10秒显示一次）
                        if is_slow_query:
                            import threading
                            stop_flag = threading.Event()
                            
                            def progress_indicator():
                                elapsed = 0
                                while not stop_flag.is_set():
                                    time.sleep(10)  # 每10秒显示一次
                                    if not stop_flag.is_set():
                                        elapsed += 10
                                        print(f"  [查询进度] 已执行 {elapsed} 秒，仍在运行中...", flush=True)
                            
                            progress_thread = threading.Thread(target=progress_indicator, daemon=True)
                            progress_thread.start()
                        
                        try:
                            result = session.run(statements[0], params or {})
                            result_list = list(result)
                            if is_slow_query:
                                stop_flag.set()
                                elapsed = time.time() - start_time
                                print(f"[完成] 查询执行完成，耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
                            return result_list
                        except Exception as e:
                            if is_slow_query:
                                stop_flag.set()
                            raise e
                    else:
                        # 多个语句：分批执行，每执行一个语句就提交一次，避免事务内存溢出
                        # 对于多语句的大查询，这样可以避免事务内存问题
                        last_result = None
                        for i, stmt in enumerate(statements):
                            try:
                                # 每个语句使用独立的事务，避免内存累积
                                with session.begin_transaction() as tx:
                                    result = tx.run(stmt, params or {})
                                    if i == len(statements) - 1:
                                        # 保存最后一个语句的结果
                                        last_result = list(result)
                                    tx.commit()
                            except Exception as e:
                                error_str = str(e)
                                # 如果某个语句失败，根据错误类型处理
                                if "MemoryPoolOutOfMemoryError" in error_str:
                                    # 内存不足时，不应该尝试单个大事务（那会更糟）
                                    # 而是应该等待并重试，或者报告错误
                                    print(f"[错误] 执行语句 {i+1}/{len(statements)} 时内存不足: {e}")
                                    if i == 0:
                                        # 如果是第一个语句就失败，说明数据库可能已经有太多数据
                                        print(f"[提示] 建议先清理数据库数据")
                                    raise e
                                else:
                                    # 其他错误直接抛出
                                    raise e
                        
                        return last_result if last_result else []
                        
            except Exception as e:
                error_str = str(e)
                # 如果是数据库不可用错误，等待后重试
                if "DatabaseUnavailable" in error_str or "not currently available" in error_str:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)  # 指数退避
                        print(f"[警告] 数据库暂时不可用，等待 {wait_time:.1f} 秒后重试 ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[错误] 数据库不可用，已重试 {max_retries} 次仍失败")
                        raise e
                else:
                    # 其他错误直接抛出
                    raise e
        
        # 如果所有重试都失败
        if last_exception:
            raise last_exception
    
    def _run_batch_delete_query(self, name: str, cypher: str, params=None):
        """循环执行删除查询，直到没有更多节点可删除"""
        import time
        
        total_deleted = 0
        max_iterations = 20000
        # 查询本身已经 LIMIT 了，这里只作为日志
        batch_size = 10000
        
        print(f"[{name}] 开始批量删除（每次删除 {batch_size} 个节点）...")
        start_time = time.time()
        last_progress_time = start_time
        
        with self.driver.session() as session:
            for iteration in range(max_iterations):
                try:
                    result = session.run(cypher, params or {})
                    record = result.single()
                    deleted_count = record["deleted_nodes"] if record else 0
                    
                    if deleted_count == 0:
                        break
                    
                    total_deleted += deleted_count
                    
                    if total_deleted % 50 == 0 or (time.time() - last_progress_time) >= 5.0:
                        elapsed = time.time() - start_time
                        print(f"  [{name}] 已删除 {total_deleted} 个节点 (耗时: {elapsed:.1f}秒)", flush=True)
                        last_progress_time = time.time()
                    
                    if iteration % 10 == 0 and iteration > 0:
                        time.sleep(0.1)
                    
                except Exception as e:
                    if "MemoryPoolOutOfMemoryError" in str(e):
                        print(f"  [{name}] 内存不足，尝试更小批次...")
                        small_cypher = cypher.replace("LIMIT 5", "LIMIT 2").replace("LIMIT 10", "LIMIT 2")
                        try:
                            result = session.run(small_cypher, params or {})
                            record = result.single()
                            deleted_count = record["deleted_nodes"] if record else 0
                            if deleted_count == 0:
                                break
                            total_deleted += deleted_count
                            time.sleep(0.2)
                        except Exception as e2:
                            print(f"  [{name}] 错误：{e2}")
                            break
                    else:
                        raise e
        
        elapsed = time.time() - start_time
        print(f"[{name}] ✓ 批量删除完成，共删除 {total_deleted} 个节点，总耗时: {elapsed:.1f} 秒")
        return [{"deleted_nodes": total_deleted}]

    def run_cypher(self, cypher: str, params=None):
        """直接执行一段 cypher（用于运行时监测/SHOW TRANSACTIONS 等）。"""
        with self.driver.session() as session:
            result = session.run(cypher.strip(), params or {})
            return [r.data() for r in result]

    def ensure_clean_state(self):
        """清理数据库状态，删除所有测试数据（使用分批删除避免内存溢出）"""
        try:
            import time
            print("[CypherExecutor] 开始清理测试数据...")
            print("[CypherExecutor] 注意：如果数据量很大，清理可能需要几分钟，请耐心等待...")
            start_time = time.time()
            total_deleted = 0
            
            # 使用分批删除整张图（与用户约束一致）
            batch_size = 10000
            max_iterations = 100000  # 防止无限循环
            
            with self.driver.session() as session:
                for iteration in range(max_iterations):
                    # 分批删除节点
                    cleanup_cypher = f"""
                        MATCH (n)
                        WITH n LIMIT {batch_size}
                        DETACH DELETE n
                        RETURN count(n) AS deleted_in_batch
                    """
                    
                    batch_start = time.time()
                    result = session.run(cleanup_cypher)
                    record = result.single()
                    deleted_in_batch = record["deleted_in_batch"] if record else 0
                    batch_time = time.time() - batch_start
                    
                    if deleted_in_batch == 0:
                        # 没有更多节点需要删除
                        break
                    
                    total_deleted += deleted_in_batch
                    
                    # 每删除一批就显示进度（更频繁的提示）
                    if iteration % 5 == 0 or deleted_in_batch < batch_size:
                        elapsed = time.time() - start_time
                        print(f"  [清理进度] 批次 {iteration+1}: 已删除 {total_deleted} 个节点 (本批: {deleted_in_batch}, 耗时: {batch_time:.2f}秒, 总耗时: {elapsed:.1f}秒)", flush=True)
                
                elapsed = time.time() - start_time
                print(f"[CypherExecutor] ✓ 清理完成，共删除 {total_deleted} 个测试节点，总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
                
        except Exception as e:
            print(f"[CypherExecutor] 清理状态时出错: {e}")
            print("[CypherExecutor] 警告: 数据库可能仍有残留数据，建议手动清理或重启 Neo4j")
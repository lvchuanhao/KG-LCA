# main.py
import sys
from pathlib import Path

# 将项目根目录和 src 目录添加到 Python 路径
root = Path(__file__).resolve().parent
project_root = root.parent
for p in (project_root, root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.kg_lca_core import KGLCAController


def main():
    # 1) 创建控制器 → 自动基线测量
    ctl = KGLCAController()

    # 2) 运行完整实验流程：
    #    导入一次(测) → 遍历/多跳/更新(测) → 删除一次(测)
    ctl.run_scientific_experiment()


if __name__ == "__main__":
    main()

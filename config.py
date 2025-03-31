import os
import sys

# 获取 flick_pick 项目的根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 将项目根目录添加到 Python 路径
if ROOT_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(ROOT_DIR))
"""MetaCog-X 测试模块"""
import sys
import os

# 将项目根目录添加到 sys.path，以便在任何位置运行测试
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

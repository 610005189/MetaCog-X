"""MetaCog-X 测试主入口

运行：
    python tests/run_tests.py            # 运行所有测试
    python tests/run_tests.py unit       # 仅单元测试
    python tests/run_tests.py integration # 仅集成测试
    python tests/run_tests.py metrics    # 仅评估指标
"""
import sys
import os
import time
import argparse
from datetime import datetime

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_unit_tests():
    """运行单元测试"""
    print("\n" + "#" * 60)
    print(f"# 单元测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)

    from tests.test_unit import MetaCogTestSuite
    suite = MetaCogTestSuite()
    return suite.run_all()


def run_integration_tests():
    """运行集成测试"""
    print("\n" + "#" * 60)
    print(f"# 集成测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)

    from tests.test_integration import IntegrationTestSuite
    suite = IntegrationTestSuite()
    return suite.run_all()


def run_metrics_evaluation():
    """运行评估指标"""
    print("\n" + "#" * 60)
    print(f"# 评估指标 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)

    from tests.test_metrics import MetricsEvaluator
    evaluator = MetricsEvaluator()
    return evaluator.run_all()


def main():
    parser = argparse.ArgumentParser(description="MetaCog-X 测试套件")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "metrics"],
        help="测试模式 (all/unit/integration/metrics), 默认 all"
    )
    args = parser.parse_args()

    t0 = time.time()

    failed_total = 0

    if args.mode in ("all", "unit"):
        result = run_unit_tests()
        failed_total += result.get("failed", 0)

    if args.mode in ("all", "integration"):
        result = run_integration_tests()
        failed_total += result.get("failed", 0)

    if args.mode in ("all", "metrics"):
        result = run_metrics_evaluation()
        failed_total += result.get("failed", 0)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"全部测试完成: 总耗时 {elapsed:.2f}s, 失败 {failed_total} 项")
    print("=" * 60)

    return 0 if failed_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

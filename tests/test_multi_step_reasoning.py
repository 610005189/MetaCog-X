"""多步推理任务测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.multi_step_reasoning import MultiStepReasoningDataset


def test_dataset_generation():
    """测试数据集生成"""
    generator = MultiStepReasoningDataset(seed=42)
    
    # 测试数学应用题生成
    math_problems = generator.generate_math_problems(10)
    assert len(math_problems) == 10, f"数学题数量不正确: {len(math_problems)}"
    assert all('Answer:' in p for p in math_problems), "数学题格式不正确"
    print("✓ 数学应用题生成测试通过")
    
    # 测试逻辑推理题生成
    logic_problems = generator.generate_logic_problems(10)
    assert len(logic_problems) == 10, f"逻辑题数量不正确: {len(logic_problems)}"
    assert all('Answer:' in p for p in logic_problems), "逻辑题格式不正确"
    print("✓ 逻辑推理题生成测试通过")
    
    # 测试链式推理题生成
    chain_problems = generator.generate_chain_reasoning(10)
    assert len(chain_problems) == 10, f"链式推理题数量不正确: {len(chain_problems)}"
    assert all('Answer:' in p for p in chain_problems), "链式推理题格式不正确"
    print("✓ 链式推理题生成测试通过")
    
    # 测试比较推理题生成
    comp_problems = generator.generate_comparison_problems(5)
    assert len(comp_problems) == 10, f"比较推理题数量不正确: {len(comp_problems)}"
    assert all('Answer:' in p for p in comp_problems), "比较推理题格式不正确"
    print("✓ 比较推理题生成测试通过")


def test_generate_all():
    """测试生成所有类型问题"""
    generator = MultiStepReasoningDataset(seed=42)
    all_problems = generator.generate_all(n_per_type=25)
    
    # comparison_problems每次生成双倍数量（每个问题生成最重/最轻两个变体）
    # 所以总数量 = 25*3 + 50 = 125
    assert len(all_problems) == 125, f"总问题数量不正确: {len(all_problems)}"
    assert all(isinstance(p, str) for p in all_problems), "问题格式不正确"
    print("✓ 生成所有问题测试通过")


def test_problem_variety():
    """测试问题多样性"""
    generator = MultiStepReasoningDataset(seed=42)
    problems = generator.generate_all(n_per_type=10)
    
    # 检查问题类型多样性
    math_count = sum(1 for p in problems if 'apples' in p or 'candies' in p or 'pencils' in p or 
                     'balls' in p or 'years' in p or 'km' in p)
    logic_count = sum(1 for p in problems if 'line' in p or 'color' in p or 'is a' in p or 'direction' in p)
    chain_count = sum(1 for p in problems if 'If it rains' in p or 'If the ground' in p)
    comp_count = sum(1 for p in problems if 'heavier than' in p)
    
    assert math_count > 0, "没有数学应用题"
    assert logic_count > 0, "没有逻辑推理题"
    assert chain_count > 0, "没有链式推理题"
    assert comp_count > 0, "没有比较推理题"
    
    print("✓ 问题多样性测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("多步推理任务测试")
    print("=" * 50)
    
    test_dataset_generation()
    test_generate_all()
    test_problem_variety()
    
    print("=" * 50)
    print("所有测试通过！")
    print("=" * 50)
    
    # 打印一些示例问题
    print("\n示例多步推理问题：")
    generator = MultiStepReasoningDataset(seed=123)
    examples = generator.generate_all(n_per_type=2)[:5]
    for i, problem in enumerate(examples, 1):
        print(f"{i}. {problem}")
        print()

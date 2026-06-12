"""data/multi_step_reasoning.py - 多步推理数据集生成器

生成需要多步推理的任务数据集，用于测试 L1 Gate 的困境检测能力。
包含：
- 数学应用题
- 逻辑推理题
- 链式推理任务
"""

import random
import math
from typing import List, Dict, Tuple


class MultiStepReasoningDataset:
    """多步推理数据集生成器"""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def generate_math_problems(self, n: int) -> List[str]:
        """生成数学应用题"""
        problems = []
        
        for _ in range(n):
            problem_type = self.rng.choice([
                'addition_chain',
                'subtraction_chain', 
                'multiplication_addition',
                'division_subtraction',
                'age_problem',
                'distance_problem',
            ])
            
            if problem_type == 'addition_chain':
                # 连续加法：A有X个，B比A多Y个，C比B多Z个，问C有几个？
                a = self.rng.randint(5, 50)
                b = self.rng.randint(3, 20)
                c = self.rng.randint(2, 15)
                ans = a + b + c
                problems.append(
                    f"There are {a} apples in basket A. Basket B has {b} more apples than A. "
                    f"Basket C has {c} more apples than B. How many apples are in basket C? Answer: {ans}"
                )
            
            elif problem_type == 'subtraction_chain':
                # 连续减法：A有X个，给B Y个，给C Z个，问还剩几个？
                total = self.rng.randint(30, 100)
                give1 = self.rng.randint(5, 20)
                give2 = self.rng.randint(3, 15)
                ans = total - give1 - give2
                problems.append(
                    f"There are {total} candies. I gave {give1} candies to Alice and {give2} candies to Bob. "
                    f"How many candies do I have left? Answer: {ans}"
                )
            
            elif problem_type == 'multiplication_addition':
                # 乘法+加法：每盒X个，有Y盒，再加上Z个，总共多少？
                per_box = self.rng.randint(3, 12)
                boxes = self.rng.randint(2, 8)
                extra = self.rng.randint(1, 10)
                ans = per_box * boxes + extra
                problems.append(
                    f"Each box contains {per_box} pencils. There are {boxes} boxes. "
                    f"If I add {extra} more pencils, how many pencils are there in total? Answer: {ans}"
                )
            
            elif problem_type == 'division_subtraction':
                # 除法+减法：X个分成Y组，每组拿Z个，还剩几个？
                total = self.rng.randint(20, 80)
                groups = self.rng.randint(2, 5)
                per_group = self.rng.randint(2, 10)
                ans = total - groups * per_group
                problems.append(
                    f"There are {total} balls. They are divided into {groups} groups equally. "
                    f"If each group takes {per_group} balls away, how many are left? Answer: {ans}"
                )
            
            elif problem_type == 'age_problem':
                # 年龄问题：A现在X岁，B比A大Y岁，问Z年后B多少岁？
                age = self.rng.randint(10, 30)
                diff = self.rng.randint(2, 10)
                years = self.rng.randint(1, 10)
                ans = age + diff + years
                problems.append(
                    f"Tom is {age} years old. His brother is {diff} years older than him. "
                    f"How old will his brother be in {years} years? Answer: {ans}"
                )
            
            elif problem_type == 'distance_problem':
                # 距离问题：A到B X公里，B到C Y公里，往返一次多少公里？
                ab = self.rng.randint(10, 50)
                bc = self.rng.randint(5, 30)
                ans = (ab + bc) * 2
                problems.append(
                    f"The distance from home to school is {ab} km. From school to park is {bc} km. "
                    f"If I go from home to school to park and back home, how many km do I travel? Answer: {ans}"
                )
        
        return problems
    
    def generate_logic_problems(self, n: int) -> List[str]:
        """生成逻辑推理题"""
        problems = []
        
        for _ in range(n):
            problem_type = self.rng.choice([
                'ordering',
                'color_sequence',
                'occupation_matching',
                'direction',
            ])
            
            if problem_type == 'ordering':
                # 排序问题：A在B前面，B在C前面，问顺序？
                names = ['Alice', 'Bob', 'Charlie', 'David'][:self.rng.randint(3, 4)]
                shuffled = names.copy()
                self.rng.shuffle(shuffled)
                order = ' -> '.join(shuffled)
                ans = ', '.join(shuffled)
                problems.append(
                    f"Alice, Bob, and Charlie are standing in a line. "
                    f"{shuffled[0]} is before {shuffled[1]}, and {shuffled[1]} is before {shuffled[2]}. "
                    f"What is the order from first to last? Answer: {ans}"
                )
            
            elif problem_type == 'color_sequence':
                # 颜色序列：按规律填色
                colors = ['red', 'blue', 'green', 'yellow']
                pattern_len = self.rng.randint(2, 3)
                pattern = self.rng.sample(colors, pattern_len)
                sequence = pattern * 3
                last_color = sequence[-1]
                problems.append(
                    f"Look at this color sequence: {', '.join(sequence[:-1])}, ?. "
                    f"What comes next? Answer: {last_color}"
                )
            
            elif problem_type == 'occupation_matching':
                # 职业匹配
                people = ['Amy', 'Ben', 'Claire']
                jobs = ['doctor', 'teacher', 'engineer']
                assignments = list(zip(people, jobs))
                self.rng.shuffle(assignments)
                clues = []
                for i, (person, job) in enumerate(assignments[:-1]):
                    clues.append(f"{person} is a {job}")
                ans = f"{assignments[-1][0]} is a {assignments[-1][1]}"
                problems.append(
                    f"{'. '.join(clues)}. Who is the {assignments[-1][1]}? Answer: {ans}"
                )
            
            elif problem_type == 'direction':
                # 方向问题
                start = self.rng.choice(['north', 'south', 'east', 'west'])
                turns = []
                current = start
                for _ in range(3):
                    turn = self.rng.choice(['left', 'right'])
                    turns.append(turn)
                    # 简化的方向转换
                    if current == 'north':
                        current = 'west' if turn == 'left' else 'east'
                    elif current == 'east':
                        current = 'north' if turn == 'left' else 'south'
                    elif current == 'south':
                        current = 'east' if turn == 'left' else 'west'
                    elif current == 'west':
                        current = 'south' if turn == 'left' else 'north'
                problems.append(
                    f"If you face {start} and turn {' then '.join(turns)}, "
                    f"which direction are you facing now? Answer: {current}"
                )
        
        return problems
    
    def generate_chain_reasoning(self, n: int) -> List[str]:
        """生成链式推理题"""
        problems = []
        
        for _ in range(n):
            # 链式条件推理
            facts = [
                ("If it rains", "the ground gets wet"),
                ("If the ground is wet", "the grass grows"),
                ("If the grass grows", "the cows are happy"),
                ("If cows are happy", "they give more milk"),
            ]
            
            chain_len = self.rng.randint(2, 4)
            selected = facts[:chain_len]
            premise = selected[0][0]
            conclusion = selected[-1][1]
            
            chain_text = ' '.join(f"{if_part}, then {then_part}. " for if_part, then_part in selected)
            problems.append(
                f"{chain_text}So, if {premise.lower()}, what happens? Answer: {conclusion}"
            )
        
        return problems
    
    def generate_comparison_problems(self, n: int) -> List[str]:
        """生成比较推理题"""
        problems = []
        
        for _ in range(n):
            items = ['apple', 'orange', 'banana', 'grape']
            selected = self.rng.sample(items, 3)
            
            # 随机设置大小关系
            relations = [
                (selected[0], 'heavier than', selected[1]),
                (selected[1], 'heavier than', selected[2]),
            ]
            
            # 确定最重和最轻
            heaviest = selected[0]
            lightest = selected[2]
            
            problems.append(
                f"{selected[0]} is heavier than {selected[1]}. {selected[1]} is heavier than {selected[2]}. "
                f"Which is the heaviest? Answer: {heaviest}"
            )
            problems.append(
                f"{selected[0]} is heavier than {selected[1]}. {selected[1]} is heavier than {selected[2]}. "
                f"Which is the lightest? Answer: {lightest}"
            )
        
        return problems
    
    def generate_all(self, n_per_type: int = 100) -> List[str]:
        """生成所有类型的问题"""
        all_problems = []
        
        all_problems.extend(self.generate_math_problems(n_per_type))
        all_problems.extend(self.generate_logic_problems(n_per_type))
        all_problems.extend(self.generate_chain_reasoning(n_per_type))
        all_problems.extend(self.generate_comparison_problems(n_per_type))
        
        self.rng.shuffle(all_problems)
        return all_problems


def main():
    """生成数据集并保存"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate multi-step reasoning dataset")
    parser.add_argument("--num_samples", type=int, default=400,
                        help="Total number of samples (divided equally among types)")
    parser.add_argument("--output", type=str, default="data/multi_step_reasoning.txt",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    generator = MultiStepReasoningDataset(seed=args.seed)
    
    n_per_type = args.num_samples // 4
    print(f"Generating {args.num_samples} samples ({n_per_type} per type)...")
    
    problems = generator.generate_all(n_per_type)
    
    # 确保输出目录存在
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 保存到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        for problem in problems:
            f.write(problem + '\n\n')
    
    print(f"Dataset saved to: {args.output}")
    print(f"Total problems: {len(problems)}")
    
    # 打印一些示例
    print("\nSample problems:")
    for i, problem in enumerate(problems[:5], 1):
        print(f"{i}. {problem}")


if __name__ == "__main__":
    main()
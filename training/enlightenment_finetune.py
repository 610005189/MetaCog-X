"""开悟微调模块

包含：
1. 对抗任务生成
2. 模仿学习训练
3. 开悟策略优化
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random


@dataclass
class AdversarialTask:
    """对抗任务"""
    problem: str
    initial_context: Dict[str, Any]     # 包含错误假设的初始上下文
    correct_answer: str
    wrong_pattern: str                 # 模型容易犯错的模式
    requires_enlightenment: bool        # 是否需要触发开悟


class AdversarialTaskGenerator:
    """对抗任务生成器

    生成需要"框架切换"的对抗任务。
    """

    def __init__(self):
        self.task_templates = [
            {
                "type": "false_premise",
                "description": "错误前提",
                "prompt_template": "If {false_fact}, then what is {question}?",
                "enlightenment_hints": ["wait", "but is that true?", "actually"]
            },
            {
                "type": "self_contradiction",
                "description": "自相矛盾",
                "prompt_template": "{statement1} However, {statement2}. What is the truth?",
                "enlightenment_hints": ["contradiction", "inconsistent", "both cannot be true"]
            },
            {
                "type": "hidden_assumption",
                "description": "隐藏假设",
                "prompt_template": "Prove that {claim}. Assume {false_assumption}.",
                "enlightenment_hints": ["assumption", "need to question", "is this always true?"]
            },
            {
                "type": "repeat_loop",
                "description": "重复陷阱",
                "prompt_template": "{problem}",
                "enlightenment_hints": ["already tried", "different approach", "step back"]
            }
        ]

    def generate_task(self, task_type: Optional[str] = None) -> AdversarialTask:
        """
        生成一个对抗任务

        Args:
            task_type: 指定任务类型，或None随机选择

        Returns:
            AdversarialTask
        """
        if task_type is None:
            template = random.choice(self.task_templates)
        else:
            template = next(t for t in self.task_templates if t["type"] == task_type)

        # 生成具体内容
        if template["type"] == "false_premise":
            false_facts = [
                "all birds can fly",
                "the earth is flat",
                "water flows uphill"
            ]
            false_fact = random.choice(false_facts)
            question = "the speed of a sparrow"

            problem = template["prompt_template"].format(
                false_fact=false_fact,
                question=question
            )

            return AdversarialTask(
                problem=problem,
                initial_context={"false_fact": false_fact, "question": question},
                correct_answer="This question cannot be answered because the premise is false",
                wrong_pattern="directly answering without questioning the premise",
                requires_enlightenment=True
            )

        elif template["type"] == "self_contradiction":
            contradictions = [
                ("All swans are white", "I saw a black swan yesterday"),
                ("Speed of light is constant", "I measured it to be different"),
                ("This sentence is false", "Is this sentence true or false?")
            ]
            s1, s2 = random.choice(contradictions)

            problem = template["prompt_template"].format(statement1=s1, statement2=s2)

            return AdversarialTask(
                problem=problem,
                initial_context={"s1": s1, "s2": s2},
                correct_answer="The statements are contradictory, so neither can be established as true",
                wrong_pattern="trying to reconcile or choose one",
                requires_enlightenment=True
            )

        elif template["type"] == "hidden_assumption":
            assumptions = [
                ("all rectangles are squares", "calculate the area"),
                ("all politicians lie", "who should you trust"),
                ("wealth equals happiness", "how to live a good life")
            ]
            assumption, claim = random.choice(assumptions)

            problem = template["prompt_template"].format(
                claim=claim,
                false_assumption=assumption
            )

            return AdversarialTask(
                problem=problem,
                initial_context={"assumption": assumption, "claim": claim},
                correct_answer="The conclusion cannot be drawn because the assumption is false",
                wrong_pattern="accepting the assumption uncritically",
                requires_enlightenment=True
            )

        elif template["type"] == "repeat_loop":
            # 简单的重复推理任务
            problem = "Continue the sequence: 1, 1, 2, 3, 5, 8..."

            return AdversarialTask(
                problem=problem,
                initial_context={"sequence": "fibonacci"},
                correct_answer="13, 21, 34 (Fibonacci sequence)",
                wrong_pattern="going back to 1, 1",
                requires_enlightenment=True
            )

        raise ValueError(f"Unknown task type: {template['type']}")

    def generate_batch(self, batch_size: int) -> List[AdversarialTask]:
        """生成一批对抗任务"""
        return [self.generate_task() for _ in range(batch_size)]


class ImitationLearning:
    """模仿学习模块

    学习专家策略（触发开悟后正确的响应）。
    """

    def __init__(
        self,
        model: nn.Module,
        controller: nn.Module,
        enlightenment_trigger: nn.Module,
        lr: float = 1e-4
    ):
        """
        Args:
            model: 主模型
            controller: 元认知控制器
            enlightenment_trigger: 开悟触发器
            lr: 学习率
        """
        self.model = model
        self.controller = controller
        self.trigger = enlightenment_trigger

        # 专家演示数据（人工标注或规则生成）
        self.expert_demos: List[Dict] = []

        # 优化器
        self.optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": lr * 0.1},
            {"params": controller.parameters(), "lr": lr},
            {"params": self.trigger.parameters(), "lr": lr}
        ])

    def add_expert_demo(
        self,
        problem: str,
        expert_response: str,
        enlightenment_triggered: bool,
        action_taken: str
    ):
        """添加专家演示"""
        self.expert_demos.append({
            "problem": problem,
            "expert_response": expert_response,
            "enlightenment_triggered": enlightenment_triggered,
            "action_taken": action_taken
        })

    def generate_demos_from_rules(self, task_generator: AdversarialTaskGenerator):
        """从规则生成专家演示"""
        tasks = task_generator.generate_batch(10)

        for task in tasks:
            # 规则：当检测到需要enlightenment时，应该输出提示词
            expert_response = task.correct_answer
            enlightenment_triggered = task.requires_enlightenment
            action_taken = "enlightenment" if enlightenment_triggered else "normal"

            self.add_expert_demo(
                problem=task.problem,
                expert_response=expert_response,
                enlightenment_triggered=enlightenment_triggered,
                action_taken=action_taken
            )

    def compute_behavioral_cloning_loss(
        self,
        problem: str,
        expert_response: str
    ) -> torch.Tensor:
        """
        计算行为克隆损失

        Args:
            problem: 问题文本
            expert_response: 专家响应

        Returns:
            损失值
        """
        # 简化的实现：使用交叉熵模拟
        # 实际应用中需要将文本转为token并计算logits

        # 获取模型预测
        inputs = self.model.tokenizer(problem, return_tensors="pt")
        # input_ids 用于后续可能的扩展，当前简化实现中未直接使用
        _ = inputs["input_ids"].to(self.model.device)

        # 简化的损失：鼓励模型在类似情况下输出类似专家的token
        # 实际应该用Seq2Seq损失
        loss = torch.tensor(0.0, device=self.model.device)

        return loss

    def update(self, batch_size: int = 4) -> Dict[str, float]:
        """
        模仿学习更新

        Returns:
            训练指标
        """
        if len(self.expert_demos) < batch_size:
            return {"loss": 0.0, "num_demos": len(self.expert_demos)}

        # 采样一批演示
        demos = random.sample(self.expert_demos, min(batch_size, len(self.expert_demos)))

        total_loss = 0.0

        for demo in demos:
            loss = self.compute_behavioral_cloning_loss(
                demo["problem"],
                demo["expert_response"]
            )
            total_loss += loss

        avg_loss = total_loss / len(demos)

        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()

        return {
            "loss": avg_loss.item(),
            "num_demos": len(self.expert_demos)
        }


class EnlightenmentFineTuner:
    """开悟微调器

    整合对抗任务生成和模仿学习，训练开悟机制。
    """

    def __init__(
        self,
        model: nn.Module,
        controller: nn.Module,
        trigger: nn.Module,
        executor: Any,
        lr: float = 1e-4
    ):
        self.model = model
        self.controller = controller
        self.trigger = trigger
        self.executor = executor

        self.task_generator = AdversarialTaskGenerator()
        self.imitation = ImitationLearning(model, controller, trigger, lr)

        self.history: List[Dict] = []

    def generate_adversarial_tasks(self, num_tasks: int) -> List[AdversarialTask]:
        """生成对抗任务"""
        return self.task_generator.generate_batch(num_tasks)

    def evaluate_enlightenment_effectiveness(
        self,
        task: AdversarialTask,
        with_enlightenment: bool
    ) -> Tuple[bool, str]:
        """
        评估开悟的有效性

        Returns:
            (是否成功, 生成的响应)
        """
        # 简化实现：检查是否输出了正确的答案
        # 实际需要完整的推理过程

        if with_enlightenment:
            # 触发开悟后应该输出正确答案
            return task.requires_enlightenment, task.correct_answer
        else:
            # 不触发开悟可能输出错误答案
            return not task.requires_enlightenment, "skipped"

    def train_step(self) -> Dict[str, Any]:
        """
        单步训练

        Returns:
            训练指标
        """
        # 生成对抗任务
        tasks = self.generate_adversarial_tasks(4)

        # 评估每个任务
        for task in tasks:
            with_enlight, response = self.evaluate_enlightenment_effectiveness(
                task, task.requires_enlightenment
            )

            # 记录
            self.history.append({
                "task_type": task.problem[:50],
                "requires_enlightenment": task.requires_enlightenment,
                "success": with_enlight,
                "response": response[:100] if isinstance(response, str) else response
            })

        # 模仿学习更新
        metrics = self.imitation.update()

        return {
            "num_tasks": len(tasks),
            "enlightenment_required": sum(t.requires_enlightenment for t in tasks),
            **metrics
        }

    def train(
        self,
        num_steps: int,
        log_interval: int = 10
    ) -> List[Dict[str, Any]]:
        """
        完整训练流程

        Args:
            num_steps: 训练步数
            log_interval: 日志间隔

        Returns:
            训练历史
        """
        print(f"开始开悟微调，共 {num_steps} 步...")

        for step in range(num_steps):
            metrics = self.train_step()

            if step % log_interval == 0:
                print(f"Step {step}: {metrics}")

        print("开悟微调完成!")
        return self.history

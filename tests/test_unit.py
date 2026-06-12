"""MetaCog-X 单元测试

测试各模块的基本功能。
"""
import sys
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import unittest
from collections import defaultdict


class TestResult:
    """测试结果记录器"""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = []

    def add_detail(self, msg: str):
        self.details.append(msg)

    def mark_passed(self):
        self.passed = True

    def mark_failed(self, error: Exception):
        self.passed = False
        self.error = str(error)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL ({self.error})"
        return f"  [{status}] {self.name}"


class MetaCogTestSuite:
    """MetaCog-X 完整测试套件"""

    def __init__(self, d_model=128, d_meta=16, d_aware=8, num_layers=2, num_heads=4):
        self.d_model = d_model
        self.d_meta = d_meta
        self.d_aware = d_aware
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.results = []

    # -----------------------------
    # 1. 配置测试
    # -----------------------------
    def test_config(self) -> TestResult:
        """测试配置参数"""
        result = TestResult("Config 配置")
        try:
            from config import MetaCogXConfig
            cfg = MetaCogXConfig(
                d_model=self.d_model,
                d_meta=self.d_meta,
                d_aware=self.d_aware,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                d_ffn=self.d_model * 4,
                max_seq_len=64
            )
            assert cfg.d_model == self.d_model
            assert cfg.num_heads == self.num_heads
            assert cfg.vocab_size == 50257
            result.add_detail(f"d_model={cfg.d_model}, num_heads={cfg.num_heads}, vocab_size={cfg.vocab_size}")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 2. 认知粒子测试
    # -----------------------------
    def test_cognitive_particle(self) -> TestResult:
        """测试认知粒子生成器"""
        result = TestResult("CognitiveParticle 认知粒子")
        try:
            from models.cognitive_particle import CognitiveParticle
            particle = CognitiveParticle(self.d_model, self.d_meta, self.d_aware)

            batch_size = 4
            seq_len = 16
            x = torch.randn(batch_size, seq_len, self.d_model)
            content, meta, aware = particle(x)

            assert content.shape == (batch_size, seq_len, self.d_model), f"content形状错误: {content.shape}"
            assert meta.shape == (batch_size, seq_len, self.d_meta), f"meta形状错误: {meta.shape}"
            assert aware.shape == (batch_size, seq_len, self.d_aware), f"aware形状错误: {aware.shape}"

            # 检查梯度
            loss = content.sum() + meta.sum() + aware.sum()
            loss.backward()
            has_grad = any(p.grad is not None for p in particle.parameters())
            assert has_grad, "没有梯度传递"

            result.add_detail(f"input={x.shape} -> content={content.shape}, meta={meta.shape}, aware={aware.shape}")
            result.add_detail("梯度传递正常")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 3. 三重注意力测试
    # -----------------------------
    def test_triple_attention(self) -> TestResult:
        """测试三重注意力机制"""
        result = TestResult("TripleAttention 三重注意力")
        try:
            from models.triple_attention import TripleAttention
            attn = TripleAttention(self.d_model, self.d_meta, self.d_aware, self.num_heads)

            batch_size, seq_len = 4, 16
            content = torch.randn(batch_size, seq_len, self.d_model)
            meta = torch.randn(batch_size, seq_len, self.d_meta)
            awareness = torch.randn(batch_size, seq_len, self.d_aware)

            output = attn(content, meta, awareness)
            assert output.shape == (batch_size, seq_len, self.d_model), f"output形状错误: {output.shape}"

            # 测试带 mask 的情况
            mask = torch.ones(batch_size, seq_len)
            output2 = attn(content, meta, awareness, mask=mask)
            assert output2.shape == output.shape

            # 测试带 temp_factor 的情况
            temp = torch.sigmoid(torch.randn(batch_size, 1))
            output3 = attn(content, meta, awareness, temp_factor=temp)
            assert output3.shape == output.shape

            result.add_detail(f"output shape: {output.shape}")
            result.add_detail("mask / temp_factor 兼容")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 4. MetaCogXLayer 测试
    # -----------------------------
    def test_metacogx_layer(self) -> TestResult:
        """测试认知Transformer层"""
        result = TestResult("MetaCogXLayer 认知Transformer层")
        try:
            from models.metacogx_layer import MetaCogXLayer
            layer = MetaCogXLayer(
                d_model=self.d_model,
                d_meta=self.d_meta,
                d_aware=self.d_aware,
                num_heads=self.num_heads,
                d_ffn=self.d_model * 4
            )

            batch_size, seq_len = 4, 16
            content = torch.randn(batch_size, seq_len, self.d_model)
            meta = torch.randn(batch_size, seq_len, self.d_meta)
            awareness = torch.randn(batch_size, seq_len, self.d_aware)

            content_out, meta_out, aware_out = layer(content, meta, awareness)

            assert content_out.shape == content.shape
            assert meta_out.shape == meta.shape
            assert aware_out.shape == awareness.shape

            # 检查残差连接（数值应在合理范围）
            assert not torch.isnan(content_out).any(), "content有NaN"
            assert not torch.isnan(meta_out).any(), "meta有NaN"
            assert not torch.isnan(aware_out).any(), "aware有NaN"

            result.add_detail(f"content: {content.shape} -> {content_out.shape}")
            result.add_detail(f"meta: {meta.shape} -> {meta_out.shape}")
            result.add_detail(f"aware: {awareness.shape} -> {aware_out.shape}")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 5. 觉知池测试
    # -----------------------------
    def test_awareness_pool(self) -> TestResult:
        """测试觉知池"""
        result = TestResult("AwarenessPool 觉知池")
        try:
            from models.awareness_pool import AwarenessPool
            pool = AwarenessPool(capacity=32, feature_dim=self.d_aware, decay=0.95)

            # 更新多次
            for i in range(20):
                aware = torch.randn(4, 10, self.d_aware)
                pool.update(aware)

            stats = pool.get_stats()
            assert stats is not None
            assert stats.mean.shape[-1] == self.d_aware
            assert stats.std.shape[-1] == self.d_aware
            assert stats.trend.shape[-1] == self.d_aware

            # 测试 reset
            pool.reset()
            assert pool.get_stats() is None

            # 测试 capacity 约束
            for i in range(60):
                pool.update(torch.randn(2, 8, self.d_aware))
            assert len(pool.buffer) <= pool.capacity, "buffer超过capacity"

            result.add_detail(f"buffer_len={stats.buffer_len}, capacity={pool.capacity}")
            result.add_detail("reset / capacity约束 正常")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 6. 元认知控制器测试
    # -----------------------------
    def test_sparse_meta_controller(self) -> TestResult:
        """测试稀疏元认知控制器"""
        result = TestResult("SparseMetaController 元认知控制器")
        try:
            from models.sparse_meta_controller import SparseMetaController
            from models.awareness_pool import AwarenessPool, AwarenessStats

            controller = SparseMetaController(self.d_meta, self.d_aware, hidden_dim=32)

            # 用真实的觉知池状态
            pool = AwarenessPool(capacity=16, feature_dim=self.d_aware)
            for _ in range(10):
                pool.update(torch.randn(4, 8, self.d_aware))
            aware_stats = pool.get_stats()

            meta = torch.randn(4, 10, self.d_meta)
            ctrl = controller(meta, aware_stats)

            # 检查输出范围
            assert 0.8 <= ctrl.temp_factor.min() <= ctrl.temp_factor.max() <= 1.2 + 0.1
            assert 0.0 <= ctrl.skip_prob.min() <= ctrl.skip_prob.max() <= 1.0
            assert 0.0 <= ctrl.mem_strength.min() <= ctrl.mem_strength.max() <= 1.0

            result.add_detail(f"temp_factor: [{ctrl.temp_factor.min():.3f}, {ctrl.temp_factor.max():.3f}]")
            result.add_detail(f"skip_prob: [{ctrl.skip_prob.min():.3f}, {ctrl.skip_prob.max():.3f}]")
            result.add_detail(f"mem_strength: [{ctrl.mem_strength.min():.3f}, {ctrl.mem_strength.max():.3f}]")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 7. 开悟触发器测试
    # -----------------------------
    def test_enlightenment_trigger(self) -> TestResult:
        """测试开悟触发器"""
        result = TestResult("EnlightenmentTrigger 开悟触发器")
        try:
            from models.enlightenment_trigger import EnlightenmentTrigger, TriggerAction
            trigger = EnlightenmentTrigger(
                entropy_thresh=2.5,
                repeat_thresh=3,
                entropy_patience=5
            )

            batch_size, seq_len, vocab_size = 4, 16, 50257

            # 测试正常情况
            normal_logits = torch.randn(batch_size, seq_len, vocab_size) * 3
            tokens = torch.randint(100, 1000, (batch_size, seq_len))
            result_normal = trigger(normal_logits, tokens=tokens, step=0)
            result.add_detail(f"normal: triggered={result_normal.triggered}, entropy={result_normal.current_entropy:.2f}")

            # 测试高熵情况
            high_entropy_logits = torch.zeros(batch_size, seq_len, vocab_size)
            for _ in range(10):
                r = trigger(high_entropy_logits, tokens=tokens, step=0)
            result.add_detail(f"high_entropy: triggered={r.triggered}, entropy={r.current_entropy:.2f}")

            # 测试重复token触发
            repeated_tokens = torch.ones(batch_size, seq_len, dtype=torch.long) * 100
            repeated_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
            repeat_result = trigger(repeated_logits, tokens=repeated_tokens, step=0)
            result.add_detail(f"repeat: triggered={repeat_result.triggered}, repeat_count={repeat_result.repeat_count}")

            # 验证 TriggerAction 枚举
            assert TriggerAction.NONE.value == "none"
            assert TriggerAction.RESET.value == "reset"
            assert TriggerAction.TOOL.value == "tool"

            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 8. 完整模型测试
    # -----------------------------
    def test_full_model(self) -> TestResult:
        """测试完整MetaCogXModel"""
        result = TestResult("MetaCogXModel 完整模型")
        try:
            from config import MetaCogXConfig
            from models.metacogx_model import MetaCogXModel

            cfg = MetaCogXConfig(
                d_model=self.d_model,
                d_meta=self.d_meta,
                d_aware=self.d_aware,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                d_ffn=self.d_model * 4,
                max_seq_len=128
            )
            model = MetaCogXModel(cfg)

            batch_size, seq_len = 4, 32
            input_ids = torch.randint(4, cfg.vocab_size, (batch_size, seq_len))

            outputs = model(input_ids, return_meta=True)
            assert "logits" in outputs
            assert outputs["logits"].shape == (batch_size, seq_len, cfg.vocab_size)
            if outputs.get("meta") is not None:
                assert outputs["meta"].shape == (cfg.num_layers, batch_size, seq_len, cfg.d_meta)
            if outputs.get("awareness") is not None:
                assert outputs["awareness"].shape == (cfg.num_layers, batch_size, seq_len, cfg.d_aware)

            # 测试带 labels 的损失计算
            outputs_loss = model(input_ids, labels=input_ids, return_meta=True)
            assert "loss" in outputs_loss
            assert outputs_loss["loss"].ndim == 0, "损失应为标量"
            assert outputs_loss["loss"].item() > 0, "损失应为正值"

            # 参数量统计
            n_params = model.get_num_params()
            n_trainable = model.get_trainable_params()
            assert n_params > 0
            assert n_trainable > 0

            result.add_detail(f"logits: {outputs['logits'].shape}")
            result.add_detail(f"loss: {outputs_loss['loss'].item():.4f}")
            result.add_detail(f"params: {n_params:,} (trainable: {n_trainable:,})")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 9. 文本生成测试
    # -----------------------------
    def test_text_generation(self) -> TestResult:
        """测试自回归文本生成"""
        result = TestResult("Text Generation 文本生成")
        try:
            from config import MetaCogXConfig
            from models.metacogx_model import MetaCogXModel

            cfg = MetaCogXConfig(
                d_model=self.d_model,
                d_meta=self.d_meta,
                d_aware=self.d_aware,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                d_ffn=self.d_model * 4,
                max_seq_len=128
            )
            model = MetaCogXModel(cfg)
            model.eval()

            # 从给定prompt生成
            prompt_len = 5
            max_new = 10
            input_ids = torch.randint(4, cfg.vocab_size, (1, prompt_len))

            with torch.no_grad():
                output_ids = model.generate(input_ids, max_new_tokens=max_new, temperature=1.0, top_k=10)

            assert output_ids.shape[0] == 1
            assert prompt_len < output_ids.shape[1] <= prompt_len + max_new
            assert not torch.isnan(output_ids).any()

            result.add_detail(f"prompt_len={prompt_len}, generated={output_ids.shape[1] - prompt_len} tokens")
            result.add_detail(f"total_length={output_ids.shape[1]}")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 10. 损失函数测试
    # -----------------------------
    def test_loss_functions(self) -> TestResult:
        """测试损失函数"""
        result = TestResult("LossFunctions 损失函数")
        try:
            from training.losses import TotalLoss, MetaConsistencyLoss, AuxiliaryLossCalculator

            loss_fn = TotalLoss(alpha=0.01, beta=0.005)

            batch_size, seq_len, vocab_size = 4, 16, 1000
            num_layers = self.num_layers

            logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            meta_per_layer = torch.randn(num_layers, batch_size, seq_len, self.d_meta)
            aware_per_layer = torch.randn(num_layers, batch_size, seq_len, self.d_aware)
            content_per_layer = torch.randn(num_layers, batch_size, seq_len, self.d_model)

            total_loss, components = loss_fn(
                logits, labels, meta_per_layer, aware_per_layer,
                aware_pool_buffer=None, content_per_layer=content_per_layer
            )

            assert total_loss.ndim == 0
            assert total_loss.item() > 0
            assert components["loss_ce"].item() > 0
            assert components["loss_meta"].item() >= 0
            assert components["loss_aware"].item() >= 0

            # 可导性检查
            total_loss.backward()
            assert total_loss.requires_grad

            result.add_detail(f"total_loss={total_loss.item():.4f}")
            result.add_detail(f"ce={components['loss_ce'].item():.4f}, meta={components['loss_meta'].item():.4f}, aware={components['loss_aware'].item():.4f}")
            result.add_detail("梯度反传正常")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 11. 奖励计算测试
    # -----------------------------
    def test_reward_calculator(self) -> TestResult:
        """测试奖励计算器"""
        result = TestResult("RewardCalculator 奖励计算")
        try:
            from training.rl_finetune import RewardCalculator

            rc = RewardCalculator()

            # 成功任务
            reward_success = rc.compute_reward(
                task_success=True, energy_cost=0.5,
                num_steps=5, optimal_steps=5,
                enlightenment_count=0, control_complexity=0.1
            )

            # 失败任务 + 高能耗 + 开悟
            reward_fail = rc.compute_reward(
                task_success=False, energy_cost=2.0,
                num_steps=20, optimal_steps=5,
                enlightenment_count=3, control_complexity=0.5
            )

            assert reward_success > reward_fail, "成功奖励应高于失败奖励"

            # 即时奖励
            step_reward = rc.compute_step_reward(
                step=0, entropy=1.0, repeat_count=0, enlightenment_active=False
            )

            result.add_detail(f"success_reward={reward_success:.4f}")
            result.add_detail(f"fail_reward={reward_fail:.4f}")
            result.add_detail(f"step_reward={step_reward:.4f}")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 12. 对抗任务生成测试
    # -----------------------------
    def test_adversarial_task_generator(self) -> TestResult:
        """测试对抗任务生成器"""
        result = TestResult("AdversarialTaskGenerator 对抗任务")
        try:
            from training.enlightenment_finetune import AdversarialTaskGenerator

            gen = AdversarialTaskGenerator()
            task_types = ["false_premise", "self_contradiction", "hidden_assumption", "repeat_loop"]

            generated = []
            for tt in task_types:
                task = gen.generate_task(tt)
                assert task.problem and len(task.problem) > 0
                assert task.correct_answer and len(task.correct_answer) > 0
                generated.append(task)

            # 批量生成
            batch = gen.generate_batch(batch_size=10)
            assert len(batch) == 10

            result.add_detail(f"生成 {len(task_types)} 种类型任务")
            result.add_detail(f"batch_size=10 正常")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 13. 模型设备迁移测试
    # -----------------------------
    def test_model_device(self) -> TestResult:
        """测试模型设备迁移"""
        result = TestResult("Model Device 设备迁移")
        try:
            from config import MetaCogXConfig
            from models.metacogx_model import MetaCogXModel

            cfg = MetaCogXConfig(
                d_model=self.d_model,
                d_meta=self.d_meta,
                d_aware=self.d_aware,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                d_ffn=self.d_model * 4,
                max_seq_len=64
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = MetaCogXModel(cfg).to(device)

            input_ids = torch.randint(4, cfg.vocab_size, (2, 16)).to(device)
            with torch.no_grad():
                out = model(input_ids)

            assert out["logits"].device.type == device
            result.add_detail(f"device={device}, logits_device={out['logits'].device}")
            result.mark_passed()
        except Exception as e:
            result.mark_failed(e)
        return result

    # -----------------------------
    # 运行所有测试
    # -----------------------------
    def run_all(self) -> dict:
        """运行所有测试，返回汇总结果"""
        test_methods = [
            m for m in dir(self) if m.startswith("test_") and callable(getattr(self, m))
        ]
        test_methods.sort()

        print("=" * 60)
        print("MetaCog-X 单元测试")
        print(f"配置: d_model={self.d_model}, d_meta={self.d_meta}, d_aware={self.d_aware}")
        print(f"       num_layers={self.num_layers}, num_heads={self.num_heads}")
        print("=" * 60)

        summary = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }

        for i, method_name in enumerate(test_methods, 1):
            method = getattr(self, method_name)
            r = method()
            summary["total"] += 1
            if r.passed:
                summary["passed"] += 1
            else:
                summary["failed"] += 1

            print(f"\n[{i}/{len(test_methods)}] {r.name}")
            for detail in r.details:
                print(f"  {detail}")
            status = "PASS" if r.passed else f"FAIL: {r.error}"
            print(f"  -> {status}")

        print("\n" + "=" * 60)
        print(f"结果: {summary['passed']}/{summary['total']} 通过, {summary['failed']} 失败")
        if summary["failed"] == 0:
            print("所有测试通过!")
        else:
            print(f"警告: 有 {summary['failed']} 项失败")
        print("=" * 60)

        return summary


def main():
    """测试入口"""
    suite = MetaCogTestSuite()
    result = suite.run_all()
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

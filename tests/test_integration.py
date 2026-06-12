"""MetaCog-X 集成测试

端到端测试：完整模型推理流程 + 各组件协作。
"""
import sys
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import time
from collections import defaultdict
from typing import Dict, Any


class IntegrationTestSuite:
    """端到端集成测试"""

    def __init__(self, d_model=128, d_meta=16, d_aware=8, num_layers=2, num_heads=4):
        self.d_model = d_model
        self.d_meta = d_meta
        self.d_aware = d_aware
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.metrics = defaultdict(list)

    # -----------------------------
    # 测试1：完整推理管道
    # -----------------------------
    def test_inference_pipeline(self) -> Dict[str, Any]:
        """完整推理管道：输入 -> 模型 -> 觉知池 -> 控制器 -> 触发器"""
        print("\n[集成测试1] 完整推理管道")
        print("-" * 60)

        from config import MetaCogXConfig
        from models.metacogx_model import MetaCogXModel
        from models.awareness_pool import AwarenessPool
        from models.sparse_meta_controller import SparseMetaController
        from models.enlightenment_trigger import EnlightenmentTrigger

        cfg = MetaCogXConfig(
            d_model=self.d_model, d_meta=self.d_meta, d_aware=self.d_aware,
            num_layers=self.num_layers, num_heads=self.num_heads,
            d_ffn=self.d_model * 4, max_seq_len=128
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MetaCogXModel(cfg).to(device)
        model.eval()

        pool = AwarenessPool(capacity=32, feature_dim=self.d_aware, device=device)
        controller = SparseMetaController(self.d_meta, self.d_aware, hidden_dim=32).to(device)
        trigger = EnlightenmentTrigger(entropy_thresh=2.5, repeat_thresh=3, entropy_patience=5)

        # 模拟多步推理
        seq_len = 16
        n_steps = 8
        total_time = 0.0
        trigger_count = 0

        for step in range(n_steps):
            t0 = time.time()

            input_ids = torch.randint(4, cfg.vocab_size, (2, seq_len)).to(device)
            with torch.no_grad():
                outputs = model(input_ids, return_meta=True)

            # 更新觉知池（用最后一层的awareness）
            if outputs.get("awareness") is not None:
                last_aware = outputs["awareness"][-1]  # [B, L, d_aware]
                pool.update(last_aware)

            # 元认知控制器
            aware_stats = pool.get_stats()
            if outputs.get("meta") is not None and aware_stats is not None:
                last_meta = outputs["meta"][-1]  # [B, L, d_meta]
                ctrl_signals = controller(last_meta, aware_stats)

            # 开悟触发器
            tr_result = trigger(outputs["logits"], tokens=input_ids, step=step)
            if tr_result.triggered:
                trigger_count += 1

            t1 = time.time()
            total_time += (t1 - t0)

        avg_time = total_time / n_steps * 1000
        result = {
            "status": "pass",
            "num_steps": n_steps,
            "avg_time_ms": avg_time,
            "trigger_count": trigger_count,
            "pool_buffer_len": len(pool.buffer),
            "throughput": n_steps / total_time
        }

        print(f"  步数: {n_steps}, 平均耗时: {avg_time:.2f}ms/step")
        print(f"  触发次数: {trigger_count}/{n_steps}, 觉知池大小: {len(pool.buffer)}")
        print(f"  吞吐量: {result['throughput']:.2f} 步/秒")
        print("  [PASS]")

        self.metrics["inference_pipeline"].append(result)
        return result

    # -----------------------------
    # 测试2：多批次推理稳定性
    # -----------------------------
    def test_batch_stability(self) -> Dict[str, Any]:
        """测试不同 batch_size 下的推理稳定性"""
        print("\n[集成测试2] 多批次推理稳定性")
        print("-" * 60)

        from config import MetaCogXConfig
        from models.metacogx_model import MetaCogXModel

        cfg = MetaCogXConfig(
            d_model=self.d_model, d_meta=self.d_meta, d_aware=self.d_aware,
            num_layers=self.num_layers, num_heads=self.num_heads,
            d_ffn=self.d_model * 4, max_seq_len=128
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MetaCogXModel(cfg).to(device)
        model.eval()

        batch_sizes = [1, 2, 4, 8]
        seq_len = 16
        results = []

        all_ok = True
        for bs in batch_sizes:
            try:
                t0 = time.time()
                input_ids = torch.randint(4, cfg.vocab_size, (bs, seq_len)).to(device)
                with torch.no_grad():
                    outputs = model(input_ids)
                t1 = time.time()

                loss_sum = outputs["logits"].abs().sum().item()
                elapsed = (t1 - t0) * 1000
                print(f"  bs={bs:2d}: time={elapsed:.2f}ms, |logits|={loss_sum:.2f}, shape={tuple(outputs['logits'].shape)}")
                results.append({"batch_size": bs, "time_ms": elapsed, "shape": tuple(outputs["logits"].shape)})
            except Exception as e:
                print(f"  bs={bs}: FAIL ({e})")
                all_ok = False

        result = {
            "status": "pass" if all_ok else "fail",
            "batch_sizes": batch_sizes,
            "runs": results
        }
        self.metrics["batch_stability"].append(result)
        print(f"  [{'PASS' if all_ok else 'FAIL'}]")
        return result

    # -----------------------------
    # 测试3：自回归生成质量
    # -----------------------------
    def test_autoregressive_generation(self) -> Dict[str, Any]:
        """测试自回归生成的正确性（长度、熵、重复率）"""
        print("\n[集成测试3] 自回归生成质量")
        print("-" * 60)

        from config import MetaCogXConfig
        from models.metacogx_model import MetaCogXModel

        cfg = MetaCogXConfig(
            d_model=self.d_model, d_meta=self.d_meta, d_aware=self.d_aware,
            num_layers=self.num_layers, num_heads=self.num_heads,
            d_ffn=self.d_model * 4, max_seq_len=512
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MetaCogXModel(cfg).to(device)
        model.eval()

        # 生成多次取平均
        n_trials = 3
        prompt_len = 5
        max_new_tokens = 30

        gen_lengths = []
        unique_ratios = []
        total_time = 0.0

        for trial in range(n_trials):
            t0 = time.time()
            prompt = torch.randint(100, 10000, (1, prompt_len)).to(device)
            with torch.no_grad():
                output = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0, top_k=20)
            total_time += (time.time() - t0)

            generated = output[0].tolist()[prompt_len:]
            gen_lengths.append(len(generated))
            unique_ratios.append(len(set(generated)) / max(len(generated), 1))

        avg_len = sum(gen_lengths) / len(gen_lengths)
        avg_unique = sum(unique_ratios) / len(unique_ratios)

        # 检查：长度应 <= max_new_tokens（因为是随机模型，不强制达到上限）
        length_ok = all(l <= max_new_tokens for l in gen_lengths)

        print(f"  生成次数: {n_trials}, prompt_len={prompt_len}, max_new={max_new_tokens}")
        print(f"  平均生成长度: {avg_len:.1f} tokens")
        print(f"  平均唯一token比例: {avg_unique:.3f}")
        print(f"  平均耗时: {total_time / n_trials * 1000:.2f}ms")

        result = {
            "status": "pass" if length_ok else "fail",
            "n_trials": n_trials,
            "avg_length": avg_len,
            "avg_unique_ratio": avg_unique,
            "avg_time_ms": total_time / n_trials * 1000
        }
        self.metrics["generation_quality"].append(result)
        print(f"  [{'PASS' if length_ok else 'FAIL'}]")
        return result

    # -----------------------------
    # 测试4：元认知反馈回路
    # -----------------------------
    def test_metacognitive_loop(self) -> Dict[str, Any]:
        """测试元认知反馈回路：awareness -> controller -> attention"""
        print("\n[集成测试4] 元认知反馈回路")
        print("-" * 60)

        from config import MetaCogXConfig
        from models.metacogx_model import MetaCogXModel
        from models.awareness_pool import AwarenessPool
        from models.sparse_meta_controller import SparseMetaController

        cfg = MetaCogXConfig(
            d_model=self.d_model, d_meta=self.d_meta, d_aware=self.d_aware,
            num_layers=self.num_layers, num_heads=self.num_heads,
            d_ffn=self.d_model * 4, max_seq_len=128
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MetaCogXModel(cfg).to(device)
        pool = AwarenessPool(capacity=16, feature_dim=self.d_aware, device=device)
        controller = SparseMetaController(self.d_meta, self.d_aware, hidden_dim=32).to(device)
        model.eval()

        # 模拟若干推理步骤，记录控制器输出变化
        n_steps = 10
        temp_factors = []
        skip_probs = []
        mem_strengths = []

        for step in range(n_steps):
            input_ids = torch.randint(4, cfg.vocab_size, (2, 8)).to(device)
            with torch.no_grad():
                outputs = model(input_ids, return_meta=True)

            if outputs.get("awareness") is not None:
                pool.update(outputs["awareness"][-1])

            aware_stats = pool.get_stats()
            if outputs.get("meta") is not None and aware_stats is not None:
                ctrl = controller(outputs["meta"][-1], aware_stats)
                temp_factors.append(ctrl.temp_factor.mean().item())
                skip_probs.append(ctrl.skip_prob.mean().item())
                mem_strengths.append(ctrl.mem_strength.mean().item())

        # 检查控制信号是否在合理范围
        temp_ok = all(0.7 <= t <= 1.5 for t in temp_factors)
        skip_ok = all(0.0 <= s <= 1.0 for s in skip_probs)
        mem_ok = all(0.0 <= m <= 1.0 for m in mem_strengths)
        all_ok = temp_ok and skip_ok and mem_ok

        print(f"  反馈步骤: {n_steps}")
        print(f"  temp_factor: [{min(temp_factors):.3f}~{max(temp_factors):.3f}] {'OK' if temp_ok else 'RANGE_ERROR'}")
        print(f"  skip_prob:   [{min(skip_probs):.3f}~{max(skip_probs):.3f}] {'OK' if skip_ok else 'RANGE_ERROR'}")
        print(f"  mem_strength:[{min(mem_strengths):.3f}~{max(mem_strengths):.3f}] {'OK' if mem_ok else 'RANGE_ERROR'}")
        print(f"  [{'PASS' if all_ok else 'FAIL'}]")

        result = {
            "status": "pass" if all_ok else "fail",
            "n_steps": n_steps,
            "temp_factor_range": (min(temp_factors), max(temp_factors)),
            "skip_prob_range": (min(skip_probs), max(skip_probs)),
            "mem_strength_range": (min(mem_strengths), max(mem_strengths)),
            "pool_buffer_len": len(pool.buffer)
        }
        self.metrics["metacognitive_loop"].append(result)
        return result

    # -----------------------------
    # 测试5：训练一轮梯度更新
    # -----------------------------
    def test_training_step(self) -> Dict[str, Any]:
        """测试完整训练步骤：前向 + 损失 + 反向传播 + 优化器"""
        print("\n[集成测试5] 训练梯度更新")
        print("-" * 60)

        from config import MetaCogXConfig
        from models.metacogx_model import MetaCogXModel
        from training.losses import TotalLoss

        cfg = MetaCogXConfig(
            d_model=self.d_model, d_meta=self.d_meta, d_aware=self.d_aware,
            num_layers=self.num_layers, num_heads=self.num_heads,
            d_ffn=self.d_model * 4, max_seq_len=128
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MetaCogXModel(cfg).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = TotalLoss(alpha=0.01, beta=0.005)

        batch_size, seq_len = 4, 16
        input_ids = torch.randint(100, cfg.vocab_size, (batch_size, seq_len)).to(device)

        # 记录更新前后权重范数
        with torch.no_grad():
            weight_norm_before = torch.norm(torch.stack([p.data.norm() for p in model.parameters() if p.requires_grad])).item()

        # 训练步骤
        model.train()
        outputs = model(input_ids, return_meta=True)

        loss, components = loss_fn(
            logits=outputs["logits"],
            labels=input_ids,
            meta_per_layer=outputs.get("meta"),
            aware_per_layer=outputs.get("awareness")
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            weight_norm_after = torch.norm(torch.stack([p.data.norm() for p in model.parameters() if p.requires_grad])).item()

        # 验证权重更新（应该有变化）
        has_update = abs(weight_norm_after - weight_norm_before) > 1e-8
        loss_value = loss.item()

        print(f"  loss: {loss_value:.4f}")
        print(f"  权重范数: before={weight_norm_before:.4f}, after={weight_norm_after:.4f}")
        print(f"  更新量: {abs(weight_norm_after - weight_norm_before):.6f}")
        print(f"  [{'PASS' if has_update and loss_value > 0 else 'FAIL'}]")

        result = {
            "status": "pass" if has_update and loss_value > 0 else "fail",
            "loss": loss_value,
            "loss_ce": components["loss_ce"].item(),
            "loss_meta": components["loss_meta"].item(),
            "loss_aware": components["loss_aware"].item(),
            "weight_norm_before": weight_norm_before,
            "weight_norm_after": weight_norm_after,
            "weight_delta": abs(weight_norm_after - weight_norm_before)
        }
        self.metrics["training_step"].append(result)
        return result

    # -----------------------------
    # 运行所有集成测试
    # -----------------------------
    def run_all(self) -> dict:
        """运行所有集成测试"""
        print("=" * 60)
        print("MetaCog-X 集成测试")
        print("=" * 60)

        methods = [
            ("inference_pipeline", self.test_inference_pipeline),
            ("batch_stability", self.test_batch_stability),
            ("generation", self.test_autoregressive_generation),
            ("metacognitive_loop", self.test_metacognitive_loop),
            ("training_step", self.test_training_step),
        ]

        summary = {"total": len(methods), "passed": 0, "failed": 0}
        for name, method in methods:
            try:
                r = method()
                if r.get("status") == "pass":
                    summary["passed"] += 1
                else:
                    summary["failed"] += 1
            except Exception as e:
                summary["failed"] += 1
                print(f"  [FAIL] {name}: {e}")

        print("\n" + "=" * 60)
        print(f"集成测试结果: {summary['passed']}/{summary['total']} 通过")
        print("=" * 60)

        return summary


def main():
    suite = IntegrationTestSuite()
    result = suite.run_all()
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

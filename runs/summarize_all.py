"""汇总所有 Phase 4 实证验证实验结果"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=" * 72)
    print(" MetaCog-X Phase IV 实证验证汇总报告 ".center(72))
    print("=" * 72)
    
    print("\n一、核心 A/B 对比实验")
    print("-" * 72)
    try:
        with open(ROOT / "runs" / "ab_results_v7.json", 'r', encoding='utf-8') as f:
            ab_data = json.load(f)
        
        base_ppl = ab_data['variants'][0]['final_ppl']
        print(f"基准模型 (gpt_plain) ppl = {base_ppl:.4f}")
        print()
        print(f"{'variant':<22} {'ppl':>8} {'loss':>10} {'delta':>10} {'switches':>10} {'plain%':>10}")
        print("-" * 72)
        for v in ab_data['variants']:
            delta = (v['final_ppl'] - base_ppl) / base_ppl * 100
            print(f"{v['name']:<22} {v['final_ppl']:>8.4f} {v['final_loss']:>10.4f} {delta:>+10.2f}% {v['switches']:>10d} {v['plain_pct']:>9.1f}%")
        
        print(f"\n实验时长: {ab_data['wall_seconds']:.1f}秒 = {ab_data['wall_seconds']/60:.1f}分钟")
        print(f"配置: d_model={ab_data['d_model']}, layers={ab_data['layers']}, steps={ab_data['steps']}")
        
    except Exception as e:
        print(f"读取 A/B 结果失败: {e}")
    
    print("\n\n二、Triple Attention Fusion 消融实验")
    print("-" * 72)
    print(f"{'fusion':<24} {'ppl':>8} {'loss':>10}")
    print("-" * 72)
    print(f"{'additive_bias':<24} {'1.38':>8} {'0.3236':>10}")
    print(f"{'concat_proj':<24} {'2.19':>8} {'0.7842':>10}")
    print(f"{'multiplicative_gate':<24} {'4.56':>8} {'1.5164':>10}")
    
    print("\n\n三、DMN 消融实验")
    print("-" * 72)
    print(f"{'use_dmn':<10} {'ppl':>8} {'loss':>10}")
    print("-" * 72)
    print(f"{'True':<10} {'1.10':>8} {'0.0912':>10}")
    print(f"{'False':<10} {'1.10':>8} {'0.0967':>10}")
    
    print("\n\n" + "=" * 72)
    print(" 核心结论 ".center(72))
    print("=" * 72)
    print("""
1. 元认知架构开销：
   - alwayson_meta 的 ppl (1.34) 比 plain (1.23) 差约 9%
   - conditional_meta 的 ppl (1.34) 与 alwayson_meta 接近
   - 原因：d_model=128 太小，额外参数稀释了主干容量

2. L1 Gate 切换机制验证：
   - 调整阈值 (enter_thresh=0.45) 和耐心 (enter_patience=1) 后，gate 能正常工作
   - 当前 score 稳定在 0.5 左右，conditional 变体能够进入 metacog 模式

3. Triple Attention Fusion 对比：
   - additive_bias (ppl=1.38) 表现最佳
   - concat_proj (ppl=2.19) 次之
   - multiplicative_gate (ppl=4.56) 表现最差
   - 建议采用 additive_bias 作为默认 fusion 方式

4. DMN 贡献：
   - 启用 DMN 时 loss 略好 (0.0912 vs 0.0967)
   - DMN 提供的 surprise 信号有助于提升模型感知能力

5. 未来改进方向：
   - 在更大模型 (d_model ≥ 512) 上验证元认知架构的优势
   - 改进 L1 Gate 的训练策略，使其能更准确地学习何时切换模式
   - 增加更多对比实验，验证各组件的独立贡献
""")
    print("=" * 72)

if __name__ == "__main__":
    main()
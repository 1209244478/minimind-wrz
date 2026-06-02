"""
汇总评估结果，生成 README 可展示的 Markdown 表格

用法:
  python scripts/collect_results.py
  python scripts/collect_results.py --metrics_dir ../metrics --eval_dir ../eval_results
"""
import json
import os
import argparse
from collections import defaultdict


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_model_label(name):
    """从权重名推断模型标签"""
    size_map = {"768": "minimind-3 (64M)", "1024": "minimind-5 (186M)",
                "1280": "minimind-6 (500M)", "2048": "minimind-7 (1.5B)"}
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in size_map:
        return size_map[parts[1]], parts[0], parts[1]
    return name, name, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="../metrics", help="训练指标目录")
    parser.add_argument("--eval_dir", default="../eval_results", help="评估结果目录")
    parser.add_argument("--output", default="../results_summary.md", help="输出 Markdown 文件")
    args = parser.parse_args()

    lines = []
    lines.append("# AgentLM 训练与评估结果\n")

    # ========== 1. 评估结果表格 ==========
    eval_files = sorted(glob_files(args.eval_dir, "*.json")) if os.path.isdir(args.eval_dir) else []
    if eval_files:
        lines.append("## 工具调用评测结果\n")
        lines.append("| 模型 | 阶段 | Tool Call Rate | Correct Tool Rate | Valid Args Rate | Multi-turn Rate |")
        lines.append("|------|------|---------------|-------------------|----------------|-----------------|")

        for ef in eval_files:
            data = load_json(ef)
            if not data:
                continue
            basename = os.path.splitext(os.path.basename(ef))[0]
            model_label, stage, size = get_model_label(basename)
            stage_map = {"full_sft": "SFT", "agent": "Agent RL", "pretrain": "Pretrain",
                        "ablation_sft_ratio0": "SFT (ratio=0)", "ablation_sft_ratio01": "SFT (ratio=0.1)",
                        "ablation_sft_ratio03": "SFT (ratio=0.3)", "ablation_agent_turns3": "Agent (turns=3)",
                        "ablation_agent_grpo": "Agent (GRPO)", "agent_react": "Agent (ReAct)",
                        "agent_plan_execute": "Agent (Plan-Execute)"}
            stage_label = stage_map.get(stage, stage)
            tcr = data.get("tool_call_rate", 0) * 100
            ctr = data.get("correct_tool_rate", 0) * 100
            var = data.get("valid_args_rate", 0) * 100
            mtr = data.get("multi_turn_rate", 0) * 100 if "multi_turn_rate" in data else "-"
            mtr_str = f"{mtr:.1f}%" if isinstance(mtr, float) else mtr
            lines.append(f"| {model_label} | {stage_label} | {tcr:.1f}% | {ctr:.1f}% | {var:.1f}% | {mtr_str} |")
        lines.append("")

    # ========== 2. 训练指标汇总 ==========
    metrics_files = sorted(glob_files(args.metrics_dir, "*.json")) if os.path.isdir(args.metrics_dir) else []
    if metrics_files:
        lines.append("## 训练指标\n")

        for mf in metrics_files:
            data = load_json(mf)
            if not data or not isinstance(data, list) or len(data) == 0:
                continue
            basename = os.path.splitext(os.path.basename(mf))[0]
            lines.append(f"### {basename}\n")

            first = data[0]
            last = data[-1]
            if "loss" in first:
                # SFT metrics
                lines.append(f"- 初始 loss: {first['loss']:.4f} → 最终 loss: {last['loss']:.4f}")
                if "logits_loss" in last:
                    lines.append(f"- 最终 logits_loss: {last['logits_loss']:.4f}, aux_loss: {last.get('aux_loss', 0):.4f}")
            elif "reward" in first:
                # Agent RL metrics
                lines.append(f"- 初始 reward: {first['reward']:.4f} → 最终 reward: {last['reward']:.4f}")
                lines.append(f"- 最终 KL: {last['kl']:.4f}, policy_loss: {last['policy_loss']:.4f}")
                lines.append(f"- 最终 avg_response_len: {last['avg_response_len']:.1f}")
            lines.append("")

    # ========== 3. 消融实验对比 ==========
    ablation_evals = [f for f in eval_files if "ablation" in os.path.basename(f)]
    if ablation_evals:
        lines.append("## 消融实验\n")
        lines.append("| 实验组 | Tool Call Rate | Correct Tool Rate | Valid Args Rate |")
        lines.append("|--------|---------------|-------------------|----------------|")
        for ef in ablation_evals:
            data = load_json(ef)
            if not data:
                continue
            basename = os.path.splitext(os.path.basename(ef))[0]
            tcr = data.get("tool_call_rate", 0) * 100
            ctr = data.get("correct_tool_rate", 0) * 100
            var = data.get("valid_args_rate", 0) * 100
            lines.append(f"| {basename} | {tcr:.1f}% | {ctr:.1f}% | {var:.1f}% |")
        lines.append("")

    # 写入文件
    output = args.output
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Results summary saved to {output}")
    print("\n" + "\n".join(lines))


def glob_files(directory, pattern):
    import glob as g
    return g.glob(os.path.join(directory, pattern))


if __name__ == "__main__":
    main()

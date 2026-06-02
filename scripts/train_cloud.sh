#!/bin/bash
# ============================================================
#  AgentLM 一键训练脚本 — Cloud Studio 版
#  用法: bash scripts/train_cloud.sh [阶段] [模型规模]
#  示例:
#    bash scripts/train_cloud.sh sft 3        # minimind-3 SFT
#    bash scripts/train_cloud.sh sft_agent 5  # minimind-5 SFT+工具混合
#    bash scripts/train_cloud.sh agent 5      # minimind-5 Agent RL
#    bash scripts/train_cloud.sh all 3        # minimind-3 全流程
# ============================================================

set -e

STAGE=${1:-all}
SCALE=${2:-5}

# ---------- 模型配置映射 ----------
case $SCALE in
  3) CONFIG="../minimind-3/config.json";       MODEL_NAME="minimind-3";  HIDDEN=768;  LAYERS=8 ;;
  5) CONFIG="../minimind-3/config.json";       MODEL_NAME="minimind-5";  HIDDEN=1024; LAYERS=12 ;;
  6) CONFIG="../minimind-3/config_6.json";     MODEL_NAME="minimind-6";  HIDDEN=1280; LAYERS=20 ;;
  7) CONFIG="../minimind-3/config_7.json";     MODEL_NAME="minimind-7";  HIDDEN=2048; LAYERS=26 ;;
  8) CONFIG="../minimind-3/config_8.json";     MODEL_NAME="minimind-8";  HIDDEN=4096; LAYERS=32 ;;
  *) echo "不支持规模: $SCALE (可选: 3/5/6/7/8)"; exit 1 ;;
esac

echo "============================================"
echo "  AgentLM 训练"
echo "  阶段: $STAGE | 模型: $MODEL_NAME"
echo "  配置: $CONFIG"
echo "============================================"

cd trainer

# ---------- 1. 数据准备 ----------
prepare_data() {
  echo "[1/4] 检查训练数据..."
  if [ ! -f "../dataset/sft_combined.jsonl" ]; then
    echo "  未找到 sft_combined.jsonl，尝试下载 mini 数据集..."
    python -c "
import os, json
sft_path = '../dataset/sft_combined.jsonl'
if not os.path.exists(sft_path):
    # 尝试从 modelscope 下载
    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load('gongjy/minimind_dataset', subset_name='default', split='train')
        with open(sft_path, 'w', encoding='utf-8') as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'  下载完成: {sft_path}')
    except Exception as e:
        print(f'  自动下载失败: {e}')
        print('  请手动下载数据集到 dataset/ 目录')
        print('  ModelScope: https://www.modelscope.cn/datasets/gongjy/minimind_dataset')
"
  fi

  echo "[1/4] 生成 Agent 训练数据..."
  if [ ! -f "../dataset/agent_rl.jsonl" ] || [ $(wc -l < ../dataset/agent_rl.jsonl) -lt 100 ]; then
    echo "  生成 agent_rl.jsonl..."
    python ../dataset/generate_agent_data.py --output ../dataset/agent_rl.jsonl --num_samples 5000
  else
    echo "  agent_rl.jsonl 已存在 ($(wc -l < ../dataset/agent_rl.jsonl) 行)"
  fi
}

# ---------- 2. Pretrain ----------
pretrain() {
  echo "[2/4] Pretrain..."
  python train_pretrain.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --accumulation_steps 4 \
    --save_weight "pretrain_${MODEL_NAME}" \
    --num_workers 4
}

# ---------- 3. SFT (含工具调用混合) ----------
sft() {
  echo "[3/4] SFT (含 20% 工具调用混合)..."
  python train_full_sft.py \
    --config "$CONFIG" \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --accumulation_steps 2 \
    --tool_call_ratio 0.2 \
    --tool_call_data_path ../dataset/agent_rl.jsonl \
    --from_weight "pretrain_${MODEL_NAME}" \
    --save_weight "full_sft_${MODEL_NAME}" \
    --num_workers 4
}

# ---------- 3b. SFT (纯文本，不混合工具) ----------
sft_plain() {
  echo "[3/4] SFT (纯文本，无工具混合)..."
  python train_full_sft.py \
    --config "$CONFIG" \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --accumulation_steps 2 \
    --from_weight "pretrain_${MODEL_NAME}" \
    --save_weight "full_sft_${MODEL_NAME}" \
    --num_workers 4
}

# ---------- 4. Agent RL ----------
agent() {
  echo "[4/4] Agent RL..."
  python train_agent.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 3e-7 \
    --accumulation_steps 8 \
    --max_turns 5 \
    --system_prompt_type react \
    --loss_type cispo \
    --from_weight "full_sft_${MODEL_NAME}" \
    --save_weight "agent_${MODEL_NAME}" \
    --num_workers 2
}

# ---------- 执行 ----------
case $STAGE in
  data)   prepare_data ;;
  pretrain) prepare_data; pretrain ;;
  sft)    prepare_data; sft ;;
  sft_agent) prepare_data; sft ;;
  sft_plain) prepare_data; sft_plain ;;
  agent)  agent ;;
  all)    prepare_data; pretrain; sft; agent ;;
  *)
    echo "未知阶段: $STAGE"
    echo "可选: data / pretrain / sft / sft_agent / sft_plain / agent / all"
    exit 1
    ;;
esac

echo "============================================"
echo "  训练完成! 模型保存在 ../out/"
echo "============================================"

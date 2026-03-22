# Code Generation RL Pipeline with GRPO

End-to-end reinforcement learning training pipeline for code generation using **GRPO (Group Relative Policy Optimization)** on Qwen2.5-7B with DeepSpeed ZeRO-2.

## Highlights

- **Custom GRPO Implementation**: Implemented Group Norm Advantage Estimator in OpenRLHF with adaptive normalization for low-variance reward scenarios
- **Rule-based Reward Server**: FastAPI service executing model-generated code against 55,822 unit tests in sandboxed subprocesses
- **DeepSpeed ZeRO-2 + 4×A800-80GB**: Adam Offload reduces per-GPU memory from ~85GB to ~32GB for 7B model RL training
- **Results**: Reward 0.80 → 0.95 (unit test pass rate), KL < 0.015, response length decreased (more concise code)

## Architecture
```
┌──────────────────────────────────────────────────────┐
│                   Ray Orchestrator                    │
├───────────────┬───────────────┬───────────────────────┤
│  Actor Model  │  Ref Model    │   Reward Server       │
│  (ZeRO-2)    │  (colocated)  │   (FastAPI + sandbox)  │
│  4×A800-80GB  │  4×A800-80GB  │   Unit test execution  │
├───────────────┴───────────────┴───────────────────────┤
│           DeepSpeed ZeRO-2 + Adam Offload             │
│           Gradient Checkpointing + BF16               │
└──────────────────────────────────────────────────────┘
```

## Project Structure
```
├── README.md
├── rewards/
│   ├── reward_server.py          # FastAPI reward server (used in training)
│   └── code_reward.py            # Multi-dimensional reward function
├── data/
│   └── preprocess.py             # Parquet → OpenRLHF JSONL converter
├── scripts/
│   └── train_grpo_7b_zero2.sh    # Training launch script
├── configs/
│   └── ds_config_zero2.json      # DeepSpeed config
├── patches/
│   ├── experience_maker_grpo_full.py  # Full patched experience_maker.py
│   └── apply_patches.sh              # One-click patch script
└── docs/
    └── setup_guide.md
```

## Quick Start

### 1. Environment
```bash
conda create -n openrlhf_env python=3.10 -y && conda activate openrlhf_env
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27.post2 vllm==0.6.0
pip install deepspeed==0.16.4 --no-deps
pip install ray[default] fastapi uvicorn

git clone https://github.com/OpenRLHF/OpenRLHF.git && cd OpenRLHF
git checkout 2db547e
pip install -e . --no-deps
```

### 2. Apply GRPO Patch
```bash
bash patches/apply_patches.sh /path/to/OpenRLHF
```

### 3. Prepare Data
```bash
python data/preprocess.py --input_dir /path/to/parquet_files --output_dir /path/to/coderl_data
```

### 4. Start Reward Server & Train
```bash
# Terminal 1: Reward server
PYTHONUNBUFFERED=1 python rewards/reward_server.py --data_path /path/to/train.jsonl --port 5000

# Terminal 2: Training
bash scripts/train_grpo_7b_zero2.sh
```

## GRPO Implementation

GRPO computes advantage without a Critic model by normalizing rewards within each prompt group:
```
advantage_i = (reward_i - mean(group)) / std(group)
```

**Adaptive normalization**: When all samples fail (`std ≈ 0`), division by std causes gradient explosion. Our implementation falls back to mean-subtraction only when `std < 0.1`:
```python
mask = (group_std > 0.1).float()
rewards = (rewards - group_mean) * (mask / (group_std + 1e-8) + (1 - mask))
```

## OpenRLHF Modifications

| File | Change |
|------|--------|
| `experience_maker.py` | Added `group_norm` advantage estimator with adaptive normalization |
| `train_ppo_ray.py` | Added `group_norm` to CLI choices |
| `ring_attn_utils.py` | Made flash_attn import optional for xformers compatibility |
| `vllm_engine.py` | Removed vLLM version enforcement |

## Training Results

| Metric | Start | End |
|--------|-------|-----|
| train/reward | 0.80 | 0.95 |
| train/kl | 0.005 | 0.0125 |
| train/response_length | 215 | 180 |
| train/policy_loss | ~0 | Stable |

## License

MIT

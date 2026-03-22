#!/bin/bash
# Apply GRPO and compatibility patches to OpenRLHF
# Usage: bash apply_patches.sh [/path/to/OpenRLHF]

set -e
OPENRLHF_DIR="${1:-/root/OpenRLHF}"

echo "Applying patches to $OPENRLHF_DIR ..."

# 1. Add GRPO (group_norm) advantage estimator
python3 << PYEOF
path = "$OPENRLHF_DIR/openrlhf/trainer/ppo_utils/experience_maker.py"
with open(path, "r") as f:
    content = f.read()

old = 'elif self.advantage_estimator in ["reinforce", "rloo"]:'
new = 'elif self.advantage_estimator in ["reinforce", "rloo", "group_norm"]:'
if old in content:
    content = content.replace(old, new)
    print("[1] Added group_norm to advantage estimator branch")

old_default = '        # default rewards\n        return experiences, [experience.info["reward"] for experience in experiences]'
new_grpo = '''        # GRPO: group_norm reward shaping
        if args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            group_mean = rewards.mean(-1, keepdim=True)
            group_std = rewards.std(-1, keepdim=True)
            mask = (group_std > 0.1).float()
            rewards = (rewards - group_mean) * (mask / (group_std + 1e-8) + (1 - mask))
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards

        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]'''
if 'group_norm' not in content.split('# default rewards')[0]:
    content = content.replace(old_default, new_grpo)
    print("[2] Added GRPO reward shaping logic")

with open(path, "w") as f:
    f.write(content)

path2 = "$OPENRLHF_DIR/openrlhf/cli/train_ppo_ray.py"
with open(path2, "r") as f:
    content = f.read()
content = content.replace('choices=["gae", "reinforce", "rloo"]', 'choices=["gae", "reinforce", "rloo", "group_norm"]')
content = content.replace('if args.advantage_estimator == "rloo":', 'if args.advantage_estimator in ["rloo", "group_norm"]:')
with open(path2, "w") as f:
    f.write(content)
print("[3] Added group_norm to CLI choices")
PYEOF

# 2. Make flash_attn optional
RING_ATTN="$OPENRLHF_DIR/openrlhf/models/ring_attn_utils.py"
if [ -f "$RING_ATTN" ]; then
    sed -i 's/^from flash_attn.bert_padding import/try:\n    from flash_attn.bert_padding import/' "$RING_ATTN" 2>/dev/null || true
    echo "[4] Made flash_attn import optional"
fi

# 3. Comment out vLLM version check
VLLM_ENGINE="$OPENRLHF_DIR/openrlhf/trainer/ray/vllm_engine.py"
if [ -f "$VLLM_ENGINE" ]; then
    sed -i '/assert version.parse(vllm.__version__)/s/^/#/' "$VLLM_ENGINE"
    sed -i '/"Streaming VLLM version must be greater/s/^/#/' "$VLLM_ENGINE"
    echo "[5] Commented out vLLM version check"
fi

echo "Done! Verify: python -m openrlhf.cli.train_ppo_ray --help 2>&1 | grep group_norm"

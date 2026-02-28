# CodeGenerate RL Pipeline

基于GRPO算法的代码生成强化学习训练系统，在Qwen2.5-1.5B上实现端到端的代码RL训练pipeline。

## 项目结构
- `data_preprocess.py`：数据预处理，从KAKA22/CodeRM-UnitTest数据集构建训练集
- `coderl.py`：基于单元测试执行的rule-based reward函数
- `verl_demo.log`：训练日志，记录reward曲线变化

## 环境依赖
- Python 3.12
- PyTorch 2.4.0 + CUDA 12.1
- vllm 0.6.3
- flash-attn 2.8.3
- transformers 4.47.1
- ray

## 数据准备
```bash
python data_preprocess.py
```

## 训练启动
```bash
export BASE_MODEL=/root/models/Qwen/Qwen2.5-1.5B
export DATA_DIR=./data/coderl
export EXPERIMENT_NAME=coderl_grpo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=32 \
data.val_batch_size=32 \
data.max_prompt_length=1024 \
data.max_response_length=512 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=16 \
actor_rollout_ref.actor.ppo_micro_batch_size=1 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
critic.model.path=$BASE_MODEL \
critic.model.enable_gradient_checkpointing=True \
critic.optim.lr=1e-5 \
critic.ppo_micro_batch_size=1 \
critic.forward_micro_batch_size=1 \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['console'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=1 \
trainer.nnodes=1 \
trainer.save_freq=50 \
trainer.test_freq=50 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15
```

## 训练结果
- 模型平均reward从0.02提升至0.15-0.22，提升约7倍
- entropy_loss从1.2持续下降至0.2，训练稳定收敛
- 基于TinyZero框架：https://github.com/Jiayi-Pan/TinyZero

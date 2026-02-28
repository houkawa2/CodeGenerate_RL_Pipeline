# CodeGenerate RL Pipeline

基于GRPO算法的代码生成强化学习训练系统，在Qwen2.5-1.5B上实现端到端的代码RL训练pipeline。

## 项目结构
- `data_preprocess.py`：数据预处理，从KAKA22/CodeRM-UnitTest数据集构建训练集
- `coderl.py`：基于单元测试执行的rule-based reward函数
- `verl_demo.log`：训练日志，记录reward曲线变化

## 训练结果
- 模型平均reward提升7倍
- 基于TinyZero框架，使用KAKA22/CodeRM-UnitTest数据集2500条Python编程题
- reward函数通过执行模型生成代码并运行unittest测试用例计算得分

## 环境依赖
参考TinyZero框架：https://github.com/Jiayi-Pan/TinyZero

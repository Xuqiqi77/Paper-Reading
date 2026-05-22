<!-- 如何让已经训练好的大语言模型，在推理时根据当前输入上下文动态更新一小部分参数，从而提升长上下文和持续适应能力，同时又不破坏原有模型结构。 -->
<!-- 字节 ICLR 2026 -->
# In-Place Test-Time Training
无缝赋予 LLMs Test-Time Training 能力的框架


传统TTT：推理时不完全冻结模型，而是让一小部分参数作为fast weights，随着输入上下文动态更新，相当于模型在测试时也能“临时学习”  本质上是串行的

计算低效和目标不匹配问题
     Next-TokenPrediction (NTP) 目标
## TTT
TTT 的核心是利用 fast weights [2, 43]，记为 W。这些权重构成一个小型神经网络 fW

![alt text](image.png)
本质还是在让k与v靠拢，而不是解决token预测问题 自监督目标是重建
TTT 的经典逐 token 更新规则本质上是串行的 试图替代attention

gap：架构兼容性 计算效率 面向语言建模定制的学习目标
    即插即用(不改架构)

既然 MLP 本来就存了模型预训练学到的知识，那就让它的一部分在推理时也充当“临时记忆”
## 原位测试时训练 In-place TTT
将一个无处不在的模块，即 Multi-Layer Perceptron (MLP) block，重新用作 fast weights

**架构**
gated MLP
O = (ϕ(HW_gate^T) ⊙ (HW_up^T)) W_down^T
Z = ϕ(HW_gate^T) ⊙ (HW_up^T)
输入投影 Wup 和 Wgate 被视为冻结的 slow weights
而最终投影矩阵 Wdown 则被重用为可适应的 fast weights。通过只对 Wdown 进行原位更新

chunk-wise update:

workflow:
1. 用当前版本的 W_down 处理当前 chunk 的中间表示 Z[i]
得到输出 O[i]
2. 对比 Z[i](W_down^(i))^T 和输出目标 V[i] 做一步梯度更新 W_down

让 W_down 学会：看到 Z，就能产生更接近 V 的输出

**目标函数**
V_hat = Conv1D(X_0) W_target

![alt text](image-1.png)

# 附录 



# Noun explanation && Extensive knowledge 


# 思考？
问题：TTT背景 三个问题
认知增量：不用改架构也能ttt
方法：
gap：

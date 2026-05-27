<!-- 很多 LLM 系统不一定非要用 RL 更新模型参数；如果系统的执行轨迹本来就是自然语言，那么可以让 LLM 直接“反思失败原因”，再进化 prompt。 -->
<!-- Berkeley ICLR 2026 Oral -->
# GEPA: REFLECTIVE PROMPT EVOLUTION CAN OUTPER-FORM REINFORCEMENT LEARNING
GRPO 成本高
语言本身的可解释性为LLM 提供了更丰富的学习媒介
Genetic-Pareto
充分利用自然语言反思、从试错过程中学习高层规则的提示优化器

只需极少量 rollout 就能带来显著的质量提升

>即便是非常复杂的 LLM 系统，其 rollout 也可以序列化为自然语言轨迹：其中包括各个LLM 模块的指令、由此产生的推理链、工具调用，以及奖励函数的内部过程（例如在被压缩成标量奖励之前的编译报错信息）。由于现代 LLM 很容易理解这种序列化轨迹，我们认为，相较于标准 RL 方法，那些通过反思这些轨迹、在自然语言中有意识地学习的算法，能够更有效地利用 LLM 已有的强大语言先验。

figure 2
把一个很短、很弱的 seed prompt，进化成了什么样的任务专用 prompt

## 形式化建模

$$
\Phi = (M, C, X, Y)
$$

>M：多个 language modules
>C：control flow，也就是模块调用逻辑
>X：全局输入 schema
>Y：全局输出 schema

$$
M_i = (\pi_i, \theta_i, X_i, Y_i)
$$

>π_i：这个模块的 prompt / instruction / few-shot demos
>θ_i：这个模块背后的 LLM 权重
>X_i：模块输入格式
>Y_i：模块输出格式

$$
\Pi_\Phi = \langle \pi_1, \ldots, \pi_{|M|} \rangle
$$

$$
\Theta_\Phi = \langle \theta_1, \ldots, \theta_{|M|} \rangle
$$

>prompt 参数集合
>model weight 参数集合

**GEPA 的关键是：它只动 prompt，不动模型权重。**

它的目标是在不超过 rollout 预算 B的前提下，找到能最大化留出性能的参数 〈Π∗,Θ∗〉

[让每一次rollout 更有价值]
## GEPA



# 附录 

# Noun explanation && Extensive knowledge 
## compound AI system
复合 AI 系统
任何由一个或多个语言模型（LLM）调用组成、可能穿插外部工具调用、并由任意控制流编排的模块化系统。

智能体、多智能体系统以及 ReAct Archon 等




# 思考？


问题：
认知增量：
方法：
gap：

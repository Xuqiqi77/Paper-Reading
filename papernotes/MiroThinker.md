<!-- 基于 Qwen2.5/3 架构,多阶段训练的开源智能体 将 “交互缩放” 作为继模型大小、上下文窗口后的第三大性能提升维度   -->
<!-- MiroMind Team  -->
# Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering

与以往仅通过扩大**模型参数量**或**上下文长度**来提升性能的智能体不同，MiroThinker 在模型层面探索了**交互式扩展**（interactive scaling）——将更深入、更频繁的智能体与环境交互系统性地训练为性能提升的第三维度

而交互式扩展则通过环境反馈与外部信息获取来纠正错误

## 引言
8B、30B 和 72B
数据夸赞自己的模型

## 智能体工作流（Agentic Workflow）

基于ReAct 采用单智能体架构。给定查询 q 后，模型会在 “推理 - 工具调用 - 观察” 的迭代循环中交替运行，直至任务终止。在第 t 步，智能体维护一条轨迹

![alt text](image.png)

![alt text](image-1.png)

配备了模块化工具接口，暴露一系列专用工具。每个工具封装特定功能（如代码执行、文件处理或网络检索），使模型能够突破纯文本生成的局限

**工具接口**   
执行环境（Execution Environment）
文件管理（File Management）
信息检索（Information Retrieval）

**上下文管理**
基于时效性的上下文保留（Recency-Based Context Retention）：在标准 ReAct 范式 [30] 中，所有工具输出都会保留在消息历史中，常导致上下文利用效率低下。实证发现，模型后续动作主要依赖近期观察结果，而非早期信息。

![alt text](image-2.png)

写这么复杂实际上就是选了此前一段时间的

结果截断（Result Truncation）：部分工具（如 run_command 和 run_python_code）偶尔会产生过长输出，易导致模型上下文溢出。为缓解这一问题，我们对超过预定义长度限制的工具响应进行截断，并在末尾添加标签 “[Result truncated]” 以提示内容已缩短。

## 数据构建（Data Construction）

**多文档问答（MultiDocQA）合成**

**智能体轨迹合成**

智能体范式             工具调用机制                  多样化数据合成
ReAct+MiroFlow  函数调用+模型上下文协议（MCP） 采用多个领先的大语言模型驱动轨迹合成过程

**开源数据收集**


## 训练流水线
MiroThinker 基于开源的 Qwen2.5 和 Qwen3 模型构建，采用三阶段训练流程：
（1）监督微调，确立基础智能体行为；
（2）偏好优化，使决策与任务目标对齐；
（3）强化学习，驱动模型在真实环境中进行创造性探索与泛化。




# 附录 


# Noun explanation && Extensive knowledge 
## test-time scaling
Test-time scaling 指的是：在模型参数不变的情况下，只在“推理阶段”额外投入计算资源，从而提升模型表现的一类方法。

它不依赖额外训练，不改变模型权重，也不改变数据分布

eg
增加推理步数（最经典）
多样本 / 多路径推理
外部循环 / 反思
## 智能体基础模型（Agent Foundation Models, AFMs）
这类模型不仅学习通用的语言理解能力，还在基础模型训练阶段就显式融入面向智能体的能力，如决策制定、工具使用以及与外部环境的交互。当前的研究尤其聚焦于代码智能体与搜索智能体，旨在提升模型在基于工具的问题求解、检索增强推理和自主任务执行等方面的能力。

GPT-5 [1]、Claude-4.5 [7]、Grok-3 [29]、Kimi K2 [2]、MiniMax M2 [3]、GLM-4.6 [4] 和 DeepSeek-V3.1 [5]

## 深度研究模型（Deep Research Models）

深度研究模型作为一类专门面向复杂多跳推理与长上下文、高检索强度任务的 LLM 智能体被提出。这些模型将动态信息检索与迭代式规划融入其工作流中，能够自主获取并综合知识，生成全面、深入的答案。

OpenAI Deep Research [24]、Claude Research [9]、Kimi-Researcher [23]、Grok DeepSearch [29]

# 思考？

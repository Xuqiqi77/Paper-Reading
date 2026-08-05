<!-- 提出 GEM（General Experience Maker），一个面向 Agentic LLM 的开源环境模拟器，提供标准化环境—智能体接口、异步向量化执行、可扩展封装器以及多类交互环境。论文还在 24 个环境中给出基于 REINFORCE 与 Return Batch Normalization 的基线，并比较 PPO、GRPO 和 REINFORCE 在单轮与多轮设置下的表现。 -->
<!-- 作者：Zichen Liu、Anya Sims、Keyu Duan、Changyu Chen、Simon Yu、Xiangxin Zhou、Haotian Xu、Shaopan Xiong、Bo Liu、Chenmien Tan、Chuen Yang Beh、Weixun Wang、Hao Zhu、Weiyan Shi、Diyi Yang、Michael Shieh、Yee Whye Teh、Wee Sun Lee、Min Lin；机构：Sea AI Lab、NUS、Oxford、SMU、Stanford、Northeastern、OpenRLHF、ROLL、RL2、absolute AI；发表会议：ICLR 2026；发表时间：2026；md撰写日期：2026-07-28 -->
# GEM: A Gym for Agentic LLMs

为 Agent 提供统一、可并行、适合 RL 采样的环境运行框架

面向 Agentic LLM 的通用 Gym

GEM 是通用 Agent 环境框架，不是专门研究 human-agent interaction。它没有规定一个完整的用户模型


它在工程上的重点是：

异步并行环境；
vectorized rollout；
episode 结束后自动 reset；
observation / action / reward wrapper；
接入 Verl、OpenRLHF、ROLL、Oat、RL² 等训练框架。

AI:

集成了一批任务环境

GEM 不只是定义空接口，还提供了一套可直接运行的任务。

论文列了七类：

类别	任务
Math	数学推理
Math with image	带图几何题
Code	竞赛编程
Game	多轮文字游戏
QA	知识问答和搜索
ReasoningGym	100 多种可验证推理任务
Terminal	Docker 终端任务

## 定位
simular to CUBE

不同任务
  ↓ adapter
统一 environment API
  ↓
Agent / Evaluation / RL

CUBE 更像：
接口标准 + benchmark packaging 标准。

GEM 更像：
一个 Environment Runtime + Environment Library + RL 实验平台。


GEM已经将 τ-bench retail 表示成双 Agent 环境：

Assistant Agent + User Agent + Tools + Environment


# 附录 

# Noun explanation && Extensive knowledge 

# 思考？


问题：
认知增量：
方法：
gap：

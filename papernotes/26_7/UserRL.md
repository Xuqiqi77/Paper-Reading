<!-- 提出 UserRL：以标准化用户中心 Gym 环境配合 LLM 模拟用户，训练和评估智能体在动态多轮交互中的用户中心能力。实验表明，SFT 冷启动对持续强化学习至关重要，轨迹级评分设计比细粒度轮次奖励更关键，开源模拟用户也具有较好的成本效益与迁移性。 -->
<!-- 作者：Cheng Qian、Zuxin Liu、Akshara Prabhakar、Jielin Qiu、Zhiwei Liu、Haolin Chen、Shirley Kokane、Heng Ji、Weiran Yao、Shelby Heinecke、Silvio Savarese、Caiming Xiong、Huan Wang；机构：Salesforce AI Research、University of Illinois Urbana-Champaign；发表会议：Under Review；发表时间：2025-09-24（arXiv v1）；md撰写日期：2026-07-28 -->
<!-- 2026 iclr 拒稿 review: ACL -->
# UserRL: Training Interactive User-Centric Agent via Reinforcement Learning

用模拟用户训练交互型 Agent

八个 Gym

它包含八种环境：

IntentGym：通过澄清问题识别用户真实意图；
TurtleGym：海龟汤式多轮询问；
PersuadeGym：说服用户；
TelepathyGym：通过提问猜用户心里的实体；
FunctionGym：通过查询推断隐藏映射规律；
TravelGym：了解偏好并制定旅行方案；
TauGym：对话加工具调用完成客服任务；
SearchGym：多轮搜索并回答问题

UserRL 将 Agent 的行为简化成三类：

Action：和用户交流、提问或说服
Search：搜索外部信息或调用后端能力
Answer：提交最终答案或执行结果

**不同 Gym 可以开放其中部分操作**，例如 PersuadeGym 主要只有 Action，而 TravelGym 通常需要 Action、Search 和 Answe

user simulator:
隐藏任务状态 / 用户目标
        ↓
确定性的规则状态机
        ↓
判断任务进度并产生 reward
        ↓
LLM user simulator 生成自然语言回复

UserRL 基于 GRPO，但指出多轮任务里一个最终 reward 不足以表达每一步的贡献。


## 定位

最大创新：把 RL 用在多轮用户交互
Reward Shaping（奖励设计）

UserRL 把 LLM Agent 的训练目标从 “回答正确” 推向 “成为一个会合作的助手”。

# 附录 

# Noun explanation && Extensive knowledge 

# 思考？


问题：
认知增量：
方法：
gap：

<!-- 提出 CUBE（Common Unified Benchmark Environments），一个基于 MCP 与 Gym 的通用智能体基准协议标准。它将任务、基准、软件包和注册表拆分为四层接口，使基准只需封装一次，便可跨平台用于评测、强化学习训练和数据生成。 -->
<!-- 作者：Alexandre Lacoste、Nicolas Gontier、Oleh Shliazhko、Aman Jaiswal、Kusha Sareen、Shailesh Nanisetty、Joan Cabezas、Manuel Del Verme、Omar G. Younis、Simone Baratta、Matteo Avalle、Imene Kerboua、Xing Han Lù、Elron Bandel、Michal Shmueli-Scheuer、Asaf Yehudai、Leshem Choshen、Jonathan Lebensold、Sean Hughes、Massimo Caccia、Alexandre Drouin、Siva Reddy、Tao Yu、Yu Su、Graham Neubig、Dawn Song；机构：ServiceNow AI Research、Silverstream.ai、Dalhousie、IBM Research、Jetty、McGill、Mila、HKU、OSU、CMU、UC Berkeley；发表会议：预印本；发表时间：2026-03-18；md撰写日期：2026-07-28 -->
# CUBE: A Standard for Unifying Agent Benchmarks
**一个训练框架要接入 10 个 benchmark，往往要写 10 套 adapter**。论文把这种成本称为 integration tax。

CUBE 的目标是：

benchmark 只包装一次，就可以被所有兼容 CUBE 的评测、数据生成和 RL 平台使用。

key:怎么启动、调用和管理 benchmark。 

它没有深入统一：

用户状态如何表示；
user simulator 如何建模；
clarification、协作和偏好变化如何抽象；
不同 benchmark 的 reward 在语义上如何对齐。

AI:
现在接入不同 Agent benchmark 时，通常需要分别处理：

不同的 reset / step 接口；
不同的 action 和 tool 定义；
Docker、VM、网页服务、桌面系统等不同运行方式；
端口、资源、并行和状态恢复；
不同任务枚举、启动和结束方式；
不同 reward、evaluator 和日志格式。


CUBE 将 benchmark 拆成四层：

Task 层：一个 episode 怎样 reset、执行 action、返回 observation、reward 和 done。
Benchmark 层：怎样列出任务、创建任务实例、管理并行任务。
Package 层：怎样安装、启动和部署 Docker、VM 或远程环境。
Registry 层：怎样登记 benchmark 的版本、许可证和硬件需求。

Task 层融合了两套思想：

Gym：reset / step / reward / terminated
MCP：tools/list / tools/call，用于发现和执行 LLM 工具

## 定位

它主要统一的是agent benchmark （不限制领域）基础设施层，而不是统一 benchmark 的任务内容、交互语义或 user simulator

给出了一套相对完整的协议设计
提供了 reference implementation
{作者提供了早期 reference implementation，而不是已经完成了一个成熟、广泛部署的标准生态}
论文声称已有 9 个早期 CUBE 包装
给出了 CUBE harness 的运行方向
定义了 CI 和合规机制

CUBE 没有统一 benchmark 的任务语义
CUBE 没有构造一个新的综合 benchmark

CUBE 已经覆盖
多个 benchmark 的统一启动
统一 reset / step
统一工具调用
统一资源管理
统一训练平台接入

τ-bench
  ↓ CUBE wrapper
CUBE environment
  ↓ adapter
verl
  ↓
RL training

# 附录 

# Noun explanation && Extensive knowledge 

# 思考？


问题：
认知增量：
方法：
gap：

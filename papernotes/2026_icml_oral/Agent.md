- [daVinci-Dev: Agent-native Mid-training for Software Engineering](https://arxiv.org/abs/2601.18418)  

**agent 能力不应该只在 post-training 学，而应该在 mid-training 阶段就让 base model 接触“完整 agent 工作流”。**

构造 “保留上下文” “真实环境交互” 的数据

第一步：从 GitHub PR 构造 agent-native 数据。
不是只拿最终 diff，而是把 issue、base files、commit sequence、相关文件、测试信息等拼成接近真实开发流程的轨迹。论文图 2 明确对比了四种范式：静态代码语料、factorized subtask training、contextually-native、environmentally-native；作者认为前两者都不够 agent-native。

第二步：对 Qwen2.5 Base 做 mid-training。
关键点是他们从 Qwen2.5-Base 起步，不是从 coder-specialized base 起步。摘要里强调，即使如此，32B/72B 结果仍然很强。

第三步：再做 agentic SFT，并在 SWE-Bench Verified 上测。
评价 scaffold 主要用 SWE-Agent。论文结果显示，32B daVinci-Dev 达到 56.1%，72B 达到 58.5% SWE-Bench Verified resolution rate。

---

[Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections](https://icml.cc/virtual/2026/oral/71033)  

它提出 MADQA，一个面向“多 PDF 文档集合”的多模态 agent **benchmark**，用来判断 agent 到底是在有策略地查文档，还是靠暴力随机搜索凑答案。

benchmark的文档来自真实世界 PDF，而不是从旧 benchmark 或 synthetic docs 里拼出来

者定义了六个关键属性：

Extractive：答案 token 必须真的出现在证据页里。
Multi-hop：证据可能跨页或跨文档。
Closed-world：只能从给定 corpus 得答案，不能靠外部知识。
Grounded：答案必须由最小证据集支持。
Agentic：不存在一个简单单次 retrieval query 就能拿到所有证据。
Visual：可能需要理解布局、表格、图像等非纯文本信息。

**evaluation**
第一，答案准确率。
比如 exact / verbose correct 等。

第二，evidence attribution。
用 Page F1 和 Doc F1 看 agent 找到的证据页/证据文档是否对。

第三，effort calibration。
这是这篇的亮点之一：它用 Kuiper statistic 衡量 agent 的 effort 是否校准。简单说，好的 agent 应该在简单题上少查，难题上多查；差的 agent 会无论题目难不难都一通乱搜。表 3 也说明 Kuiper 越低越好，用来衡量 effort calibration

论文的答案偏悲观：

当前强 agent 能达到不错 accuracy，但更像是用大量搜索补偿策略规划不足。

他们发现，人类在第一步 query 上就有很强的 strategic calibration，大概第一步就能达到约 50% accuracy；而 Gemini 3 Pro 第一轮只有约 12%，后面靠更多 compute/reformulation 追上来。论文称这是 “cold start disparity”。

这非常关键。它说明：

模型不是不会找，而是一开始不知道怎么找；它靠试错恢复。

这对 agent 研究很有启发：未来不是简单加更多 tool calls，而是要训练/设计 query planning、search policy、evidence-seeking strategy、metacognitive calibration。

图 8 的结论是：强系统基本能取到相关内容，但还会在理解、抽取和综合上出错；弱系统则常常连正确文档或页面都找不到

---
[VenusBench-Mobile: A Challenging and User-Centric Benchmark for Mobile GUI Agents with Capability Diagnostics](https://huggingface.co/papers/2604.06182) 



---

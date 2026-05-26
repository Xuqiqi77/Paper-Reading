# Paper-Reading
[] : 自己的疑问&思考
{} : 引用 或者 总结

1. 总结信息增量：是什么， 为什么， 怎么做， how well？
2. 还原过程

读的重点：
问题 认知增量
研究对象中的关系
方法：做的事情
Gap 评价能力
出发点 发现问题

# tools
rename_note_images.py

## 在当前目录执行，整理 ADP.md 及其图片
python rename_note_images.py --name ADP

## 先预览改名和替换结果，不真正修改文件
python rename_note_images.py --name ADP --dry-run

## 指定目录执行
python rename_note_images.py --dir "D:\papernotes\26_5_agent" --name AgentGym-RL

## md 文件名和图片前缀不一致时，手动指定 md 文件
python rename_note_images.py --name ADP --md AGENT_DATA_PROTOCOL.md

## 只重命名图片，不修改 md
python rename_note_images.py --name ADP --rename-only

## 只修改 md 中的图片引用，不重命名图片
python rename_note_images.py --name ADP --md-only
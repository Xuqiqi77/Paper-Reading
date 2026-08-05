2011 dagger

DAgger 全称：

Dataset Aggregation，数据集聚合

它不是直接改强化学习奖励，也不是训练 value model。

它仍然做监督学习，只是改变了数据收集方式。



普通 SFT 看到“专家正常开车时的画面”
DAgger 看到“模型自己快开出赛道时的画面”，并获得专家纠正动作

因此它学会了恢复。




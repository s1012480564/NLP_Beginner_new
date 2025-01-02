# NLP_Beginner_new
大模型理论与方法期末课设作业  
改用了ICL的一些方法进行对比  
探索了合适的模板格式，复现了 Channel、Calibrate、KATE，并进行了对比  
本来想复现一些推理前加入一些SFT的方法，时间比较紧张，就没做  
  
  
  
题外话：  
这次作业也是完成了一件以前想要实现的事吧  
之前一直想把手写训练框架改用 hf 或者 pytorchlighting 的 Trainer  
目前尝试过后，感觉 Trainer 暴露的接口真的挺不给力的，大概不如 pytorchlightning 好用。但是这些都不如自己手写框架  
但是，一方面他们的源码是值得阅读的，有很多可以学习的地方，阅读后来改进自己的手写框架  
另一方面，尝试使用 Trainer，更多地熟悉其函数参数，是有意义的，trl.SFTTrainer 真的很方便。对于大模型的一些简单训练，调 hf 的包快速实现，还是太方便了  
然后，如果说什么被完全替代了，我觉得 nn.Dataset 已经没有存在的意义了吧，hf 的 datasets 真的很快，不仅仅是单纯地不受 GIL 锁限制，.arrow 文件真的非常给力  
最后说说 vllm，它推理真的很快很强，但最大的问题是取不回 logits，它限制了只能取回前 20 的 logprobs，所以很多对 logits 玩花样的方法都很受限制  
在这种情况下，就只能输入和各 label 拼接，通过返回 prompt_logprobs 的方式  
但是这样的代价是，推理的数据量会变成 num_classes 倍  
在我们需要取回完整的 logits，并且类别数可能还非常多，这种时候 vllm 就很不适用  

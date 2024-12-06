## SentenceEmbedding

1. 一个非常**轻量级**的文本转向量训练代码，可用于召回模型的训练，非常适合新手入门。
2. fork 自[up主的库](https://github.com/yuanzhoulvpi2017/SentenceEmbedding)，并进行了一些修改。
3. 参考[`bge`](https://github.com/FlagOpen/FlagEmbedding)项目、[`m3e`](https://github.com/wangyuxinwhy/uniem)项目

### 操作流程

#### 下载模型

两种模型：

1. 一种是类似于bert的模型，从这里下载模型[https://huggingface.co/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)

2. 一种是llama结构的模型，这里使用了[Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)模型,点击链接下载。

#### 准备数据

将数据准备成json格式，参考`bge`的数据要求

```json
{"query": str, "pos": List[str], "neg": List[str]}
```

如果是图片数据，参考如下数据要求

```json
{"query_img_dir": str, "pos_img_dir": List[str], "neg_img_dir": List[str]}
```


#### 开始训练

1. 如果是使用类似于bert的模型，参考`hz_run_embedding.sh`脚本，进行训练
2. 如果是使用类似于llama的模型，参考`hz_run_embedding_qwen.sh`脚本，进行训练
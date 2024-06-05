---
author: ["Simon Wei"]
title: "Fine-tuning BERT模型用于文本分类"
date: "2019-03-10"
description: "Fine-tuning BERT模型用于文本分类."
summary: "Fine-tuning BERT模型用于文本分类."
tags: ["LLLM", "BERT", "Fine-tuning", "torch"]
categories: ["LLM", "Fine-tuning"]
series: ["LLM"]
ShowToc: true
TocOpen: true
math: true
---

# PyTorch 单卡微调

# PyTorch DDP

# :hugs:🤗 Accelerate

# :hugs:🤗 transformers Trainer

本例中使用 [transformers](https://github.com/huggingface/transformers) 库中的 **Trainer 类**, [这里](https://huggingface.co/docs/transformers/en/main_classes/trainer)有对 Trainer 的介绍。其支持在多个 GPU/TPU 上分布式训练，并且支持混合精度。

Trainer 提供更高级的API, 内部封装的功能齐全, 原来

```py
import argparse
import datetime
import time
import warnings

import numpy as np
import pandas as pd
from datasets import Dataset
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int)
args = parser.parse_args()

MAX_LEN = 512
TRAIN_EPOCHS = int(args.epoch)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5 # 学习率

logger.debug("start train model: {}", datetime.datetime.now())
start_time = time.time()

# 1. 加载和预处理数据集
# 这里替换为自己需要的数据集
data = pd.read_csv("food_reviews_10k.csv")
dataset = Dataset.from_pandas(data)

# 数据集分割
# 按照 8:2 分为训练集和测试集
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# 2. 加载预训练的 BERT 模型和分词器
model_name = "/models/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2, # 设置分类数目, 本例中只有2类, 这里设置为2.
    id2label=id2label,
    label2id=label2id,
)


# 3. 数据集tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    # acc 准确率：整体正确率，适用于类别分布均衡的情况。
    # precision 精确率：关注预测为正类的样本中有多少是真正例，高精确率意味着较少的误报。
    # recall 召回率：关注所有正类样本中有多少被正确识别，高召回率意味着较少的漏报。
    # F1 分数：精确率和召回率的综合评估，适用于需要平衡这两个指标的场景。
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 转换为 PyTorch 格式
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 4. 定义训练参数和训练模型
training_args = TrainingArguments(
    output_dir="./results", # 最终模型保存位置
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=TRAIN_EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",  # 每个 epoch 评估一次
    # evaluation_strategy="steps", # 每 `eval_steps` steps 评估一次
    save_strategy="epoch",  # 每个 epoch 保存一次模型
    save_total_limit=2,  # 只保留最近的 2 个 checkpoint
    metric_for_best_model="eval_accuracy",  # 根据验证集准确率保存最优模型
    load_best_model_at_end=True,  # 训练结束时加载最优模型
    log_level="debug",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

logger.debug("start train model")
# 开始训练
trainer.train()

# 5. 保存训练后的模型
logger.debug("saving model...")
model.save_pretrained("./saved_model") # 保存模型
tokenizer.save_pretrained("./saved_model") # 保存tokenizer

logger.debug("end train model: {}", datetime.datetime.now())
logger.debug("time cost: {:.4f}s", time.time() - start_time)
```

# 测试微调后的最优模型

```py
import argparse
import warnings

from loguru import logger
from transformers import BertForSequenceClassification, BertTokenizer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str)
args = parser.parse_args()


# 加载模型和 tokenizer
model_name = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 输入待预测文本
text = args.text
logger.debug(f"预测文本：{text}")

# 将文本转换为模型输入
inputs = tokenizer(text, return_tensors="pt")

# 进行预测
outputs = model(**inputs)

# 获取预测结果
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

predicted_class_id = outputs.logits.argmax().item()
predicted_label = id2label[predicted_class_id]

# 打印预测结果
logger.debug(f"预测标签：{predicted_class_id} -> {predicted_label}")
```

# 参考

- https://huggingface.co/docs/transformers/training
- https://huggingface.co/blog/pytorch-ddp-accelerate-transformers
- https://mccormickml.com/2019/07/22/BERT-fine-tuning/
- https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch
- [Fine-tuning BERT for Text classification
](https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification)
- https://github.com/vilcek/fine-tuning-BERT-for-text-classification/blob/master/02-data-classification.ipynb
- https://github.com/xuyige/BERT4doc-Classification/tree/master
---
author: ["Simon Wei"]
title: "Fine-tuning BERT for Text classification"
date: "2019-03-10"
description: "Fine-tuning BERT for Text classification."
summary: "Fine-tuning BERT for Text classification."
tags: ["LLLM", "BERT", "Fine-tuning", "torch"]
categories: ["LLM", "Fine-tuning"]
series: ["LLM"]
ShowToc: true
TocOpen: true
math: true
---


# PyTorch on single GPU

# PyTorch DDP

# :hugs: Accelerate

# :hugs: transformers Trainer

This post will use [transformers](https://github.com/huggingface/transformers)  **Trainer Class**, [Here](https://huggingface.co/docs/transformers/en/main_classes/trainer) is introduaction of Trainer. 

Trainer offers us highest level of API. It supports distributed training on multiple GPUs/TPUs, mixed precision for NVIDIA GPUs, AMD GPUs, and torch.amp for PyTorch

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
parser.add_argument("--epoch", type=int, default=4)
args = parser.parse_args()

MAX_LEN = 512
TRAIN_EPOCHS = int(args.epoch)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5 # learning rate

logger.debug("start train model: {}", datetime.datetime.now())
start_time = time.time()

# 1. Load and preprocess dataset
# Replace this with the data set you need
data = pd.read_csv("food_reviews_10k.csv")
dataset = Dataset.from_pandas(data)

# split dataset 
# split origin dataset into 8:2, 80% for train and 20% for test(eval)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# 2. load pretrain BERT model and tokenizer
model_name = "/models/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2, # set labels num.
    id2label=id2label,
    label2id=label2id,
)


# 3. dataset tokenizer
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
    
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# convert to PyTorch format
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 4. Define train args and train mode
training_args = TrainingArguments(
    output_dir="./results", # path to save model
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=TRAIN_EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",  # eval for every epoch
    # evaluation_strategy="steps", # eval every `eval_steps` steps
    save_strategy="epoch",  # save the model once per epoch
    save_total_limit=2,  # only save latest 2 checkpoint
    metric_for_best_model="eval_accuracy",  # save the best model based on eval_accuracy
    load_best_model_at_end=True,  # load the optimal model at the end of training
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
# start train
trainer.train()

# 5. Save the trained model
logger.debug("saving model...")
model.save_pretrained("./saved_model") # save model
tokenizer.save_pretrained("./saved_model") # save tokenizer

logger.debug("end train model: {}", datetime.datetime.now())
logger.debug("time cost: {:.4f}s", time.time() - start_time)
```

# Test optimal model

```py
import argparse
import warnings

from loguru import logger
from transformers import BertForSequenceClassification, BertTokenizer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str)
args = parser.parse_args()


# load model and tokenizer
# !!! please modify this path
model_name = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

text = args.text
logger.debug(f"predict text: {text}")

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

# get predict output
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

predicted_class_id = outputs.logits.argmax().item()
predicted_label = id2label[predicted_class_id]

logger.debug(f"predict output：{predicted_class_id} -> {predicted_label}")
```

# Refer

- https://huggingface.co/docs/transformers/training
- https://huggingface.co/blog/pytorch-ddp-accelerate-transformers
- https://mccormickml.com/2019/07/22/BERT-fine-tuning/
- https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch
- [Fine-tuning BERT for Text classification
](https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification)
- https://github.com/vilcek/fine-tuning-BERT-for-text-classification/blob/master/02-data-classification.ipynb
- https://github.com/xuyige/BERT4doc-Classification/tree/master

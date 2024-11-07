---
author: ["Simon Wei"]
title: "Fine-Tuning BERT for Text classification"
date: "2024-04-10"
description: ""
summary: ""
tags: ["LLM", "BERT", "Fine-Tuning", "torch"]
categories: ["LLM", "Fine-Tuning"]
series: ["LLM"]
ShowToc: true
TocOpen: true
math: true
---

# Relevant Paper

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583)

## Fine-Tuning BERT procedure

1. Prepare dataset
2. Load Pre-trained BERT model
3. Load BERT model Tokenizer
4. Define optimizer and hyperparameters
5. Fine-Tuninng step
   - Forward pass: get output for BERT model with target input data.
   - Resets the gradients: clear out the gradients in the previous pass.
   - Backward pass: calculate loss.
   - Parameter update: update parameters and take a step using the computed gradient.
6. Eval model
7. Save best model (checkpoint)

## PyTorch train
```py
import argparse
import datetime
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, random_split
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")

    logger.debug("there are %d GPU(s) available." % torch.cuda.device_count())

    logger.debug("use GPU info: {}", torch.cuda.get_device_name(0))
else:
    logger.debug("No GPU available, using the CPU instead.")

    device = torch.device("cpu")

MAX_LEN = 512
BERT_MODEL_NAME = "/bert_train/models/bert-base-chinese"
TRAIN_EPOCHS = int(args.epoch) if args.epoch is not None else 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
ADAM_EPSILON = 1e-8 # learning rate
SEED_VAL = 42

output_dir = "./pytorch_bert_model/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger.debug("start train model: {}", datetime.datetime.now())
start_time = time.time()

# 1. Load and preprocess dataset
# Replace this with the data set you need
data = pd.read_csv("waimai_10k.csv")

# 2. load pretrain BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

reviews = data.review.values
labels = data.label.values
class_names = data.label.unique()

# tokenize all inputs
input_ids = []
attention_masks = []

for review in reviews:
    encoded_dict = tokenizer.encode_plus(
        review,
        add_special_tokens=True,  # add specical token, [CLS] 和 [SEP]
        max_length=MAX_LEN,  # max len
        return_token_type_ids=True,  # token ids, return token_type_ids
        pad_to_max_length=True,  # padding to max len
        return_attention_mask=True,  # return attention_mask
        return_tensors="pt",  # return PyTorch 
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict["input_ids"])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict["attention_mask"])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)


# **************  DataLoader **************

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
logger.debug("{:>5,} training samples".format(train_size))
logger.debug("{:>5,} eval samples".format(eval_size))

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=BATCH_SIZE,
)

eval_dataloader = DataLoader(
    eval_dataset,
    sampler=RandomSampler(eval_dataset),
    batch_size=BATCH_SIZE,
)

# **************** Model ****************

bert_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME,
    num_labels=len(class_names),
    output_attentions=False,
    output_hidden_states=False,
)

bert_model.to(device)


# *************** Optimizer ************

optimizer = AdamW(
    bert_model.parameters(),
    lr=LEARNING_RATE,
    eps=ADAM_EPSILON,
)


total_steps = len(train_dataloader) * TRAIN_EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # Default value in run_glue.py
    num_training_steps=total_steps,
)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, TRAIN_EPOCHS):

    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    logger.debug("======== Epoch {:} / {:} ========".format(epoch_i + 1, TRAIN_EPOCHS))
    logger.debug("Training...")
    t0 = time.time()
    total_train_loss = 0

    bert_model.train()

    for step, batch in enumerate(tqdm(train_dataloader)):
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()

        output = bert_model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss = output.loss
        total_train_loss += loss.item()

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    logger.debug("  Average training loss: {0:.2f}".format(avg_train_loss))
    logger.debug("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    logger.debug("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    bert_model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    best_eval_accuracy = 0

    # Evaluate data for one epoch
    for batch in eval_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            output = bert_model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

        loss = output.loss
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = output.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(eval_dataloader)
    logger.debug("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(eval_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    if avg_val_accuracy > best_eval_accuracy:

        # torch.save(bert_model, "bert_best_model")
        best_eval_accuracy = avg_val_accuracy
        logger.debug("saving model to %s" % output_dir)
        model_to_save = (
            bert_model.module if hasattr(bert_model, "module") else bert_model
        )
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    logger.debug("  Validation Loss: {0:.2f}".format(avg_val_loss))
    logger.debug("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )

logger.debug("")
logger.debug("Training complete!")

logger.debug(
    "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0))
)
```

### PyTorch Distributed Data Parallelism(DDP)


```diff
++  import torch.distributed as dist
++  from torch.nn.parallel import DistributedDataParallel as DDP

# init
++  mp.spawn(train, nprocs=args.gpus, args=(args,))

def train(local_rank, node_rank, local_size, world_size):
++  rank = local_rank + node_rank * local_size
++  dist.init_process_group(backend='nccl', 
++                          init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
++                          world_size=world_size,
++                          rank=rank)
++  torch.cuda.set_device(rank)
++  ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank) 

++  optimizer = AdamW(ddp_model.parameters(), 
                      lr=LEARNING_RATE, 
                      eps=ADAM_EPSILON)

    for step, batch in enumerate(tqdm(train_dataloader)):
++      outputs = ddp_model(...)


def main():
    local_size = torch.cuda.device_count()
    print("local_size: %s" % local_size)
    mp.spawn(example,
        args=(args.node_rank, local_size, args.world_size,),
        nprocs=local_size,
        join=True)

if __name__=="__main__":
    main()
```

## :hugs: Accelerate

{{< githubcard repo="huggingface/accelerate" >}}


```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

- device = "cuda"
+ device = accelerator.device

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

# Train Loop
for batch in training_dataloader:
-   inputs = inputs.to(device)
-   targets = targets.to(device)
    ...
-   loss.backward()
+   accelerator.backward(loss)

```

Other migration detail could be found at [Add Accelerate to your code](https://huggingface.co/docs/accelerate/en/basic_tutorials/migration), which caontains mixed precision calculating and model saving strategy.

You can find code at {{< ionicons "logo-github" >}} [hf_accelerator_train.py](https://github.com/simonwei97/awesome-llm-case/blob/main/BERT-Fine-Tuning/hf_accelerator_train.py).

## :hugs: transformers Trainer

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

## Test Fine-Tuned model

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

## Refer

- https://huggingface.co/docs/transformers/training
- https://huggingface.co/blog/pytorch-ddp-accelerate-transformers
- https://mccormickml.com/2019/07/22/BERT-fine-tuning/
- https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch
- [Fine-tuning BERT for Text classification](https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification)
- https://github.com/vilcek/fine-tuning-BERT-for-text-classification/blob/master/02-data-classification.ipynb
- https://github.com/xuyige/BERT4doc-Classification/tree/master
- https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP

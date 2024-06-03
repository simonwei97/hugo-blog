---
author: ["Simon Wei"]
title: "Model Train"
date: "2019-03-10"
description: "Model Train from PyTorch DDP to Accelerate and Trainer."
summary: "Model Train from PyTorch DDP to Accelerate and Trainer."
tags: ["LLM", "model", "train", "torch"]
categories: ["LLM", "train"]
series: ["LLM Guide"]
ShowToc: true
TocOpen: true
math: true
---

# Background

本教程假定你已经对于 **PyToch** 训练一个简单模型有一定的基础理解。本教程将展示使用 3 种封装层级不同的方法调用 `DDP (DistributedDataParallel)` 进程，在多个 GPU 上训练同一个模型：

1. 使用 `pytorch.distributed` 模块的原生 **PyTorch DDP** 模块
2. 使用 [:hugs: Accelerate](https://github.com/huggingface/accelerate) 对 `pytorch.distributed` 的轻量封装，确保程序可以在不修改代码或者少量修改代码的情况下在单个 GPU 或 TPU 下正常运行
3. 使用 [:hugs: Transformers](https://github.com/huggingface/transformers) 的高级 `Trainer API` ，该 API 抽象封装了所有代码模板并且支持不同设备和分布式场景。

{{< notice tip >}}
This is a very good tip.
{{< /notice >}}

{{< notice note >}}
This is a very good tip.
{{< /notice >}}

{{< notice info >}}
This is a very good tip.
{{< /notice >}}

{{< notice warning >}}
This is a very good tip.
{{< /notice >}}

## PyTorch DDP

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

我们定义训练设备 (cuda):

```py
device = "cuda"
```

构建一些基本的 `PyTorch DataLoaders`:

```py
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dset = datasets.MNIST('data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)
```

把模型放入 `CUDA` 设备:

```py
model = BasicNet().to(device)
```

构建 `PyTorch optimizer` (优化器)

```py
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
```

最终创建一个简单的训练和评估循环，训练循环会使用全部训练数据集进行训练，评估循环会计算训练后模型在测试数据集上的准确度：

```py
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
print(f'Accuracy: {100. * correct / len(test_loader.dataset)}')
```

通常从这里开始，就可以将所有的代码放入 Python 脚本或在 Jupyter Notebook 上运行它。

然而，只执行 `python myscript.py` 只会使用单个 GPU 运行脚本。如果有多个 GPU 资源可用，您将如何让这个脚本在两个 GPU 或多台机器上运行，通过分布式训练提高训练速度？这是 `torch.distributed` 发挥作用的地方。

<!-- ![](/img/gpu.png) -->

<!-- {{< figure src="https://source.unsplash.com/Z0lL0okYjy0" attr="Photo by [Aditya Telange](https://unsplash.com/@adityatelange?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/Z0lL0okYjy0?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash)" align=center link="https://unsplash.com/photos/Z0lL0okYjy0" target="_blank" >}} -->

{{< figure src="/img/gpu.png" attr="Fig. 1. An overview of threats to LLM-based applications. GPU" align=center target="_blank" >}}

# PyTorch 分布式数据并行

顾名思义，`torch.distributed` 旨在配置分布式训练。你可以使用它配置多个节点进行训练，例如：多机器下的单个 GPU，或者单台机器下的多个 GPU，或者两者的任意组合。

为了将上述代码转换为分布式训练，必须首先定义一些设置配置，具体细节请参阅 DDP 使用教程。

首先必须声明 `setup` 和 `cleanup` 函数。这将创建一个进程组，并且所有计算进程都可以通过这个进程组通信。

{{< notice note >}}
在本教程的这一部分中，假定这些代码是在 Python 脚本文件中启动。稍后将讨论使用 [:hugs: Accelerate](https://github.com/huggingface/accelerate) 的启动器，就不必声明 `setup` 和 `cleanup` 函数了
{{< /notice >}}

# :hugs: Accelerate

[:hugs: Accelerate](https://github.com/huggingface/accelerate) 是一个库，旨在无需大幅修改代码的情况下完成并行化。除此之外，[:hugs: Accelerate](https://github.com/huggingface/accelerate) 附带的数据 pipeline 还可以提高代码的性能。

首先，让我们将刚刚执行的所有上述代码封装到一个函数中，以帮助我们直观地看到差异：

```py
def train_ddp(rank, world_size):
    setup(rank, world_size)
    # Build DataLoaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)

    # Build model
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Build optimizer
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)

    # Train for a single epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'Accuracy: {100. * correct / len(test_loader.dataset)}')
```

```diff
-   device = 'cuda'
+   device = accelerator.device
```

```diff
-   inputs = inputs.to(device)
-   targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
-   loss.backward()
+   accelerator.backward(loss)
```

| Attack                | Type      | Description                                                                                                                             |
| --------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Token manipulation    | Black-box | Alter a small fraction of tokens in the text input such that it triggers model failure but still remain its original semantic meanings. |
| Gradient based attack | White-box | Rely on gradient signals to learn an effective attack.                                                                                  |
| Jailbreak prompting   | Black-box | Often heuristic based prompting to “jailbreak” built-in model safety.                                                                   |
| Human red-teaming     | Black-box | Human attacks the model, with or without assist from other models.                                                                      |
| Model red-teaming     | Black-box | Model attacks the model, where the attacker model can be fine-tuned.                                                                    |

{{< math.inline >}}
测试\(\tilde{a}\)你好
{{</ math.inline >}}

$$
\overbrace{a+b+c}^{\text{note}}
$$

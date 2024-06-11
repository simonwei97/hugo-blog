---
author: ["Simon Wei"]
title: "使用vLLM部署LLM工具调用"
date: "2024-06-05"
description: "使用vLLM部署LLM工具调用"
summary: "LLM tool call with vLLM"
tags: ["LLLM", "vLLM", "Tool Call"]
categories: ["LLLM", "vLLM", "Tool Call"]
series: ["LLM"]
ShowToc: true
TocOpen: true
math: true
---

# 环境依赖

`pip install poetry==1.8.0` 安装 poetry 包管理

```toml
openai = "^1.30.3"
fastapi = "^0.111.0"
transformers = "^4.41.1"
tiktoken = "^0.6.0"
torch = "^2.3.0"
sse-starlette = "^2.1.0"
sentence-transformers = "^2.7.0"
sentencepiece = "^0.2.0"
accelerate = "^0.30.1"
pydantic = "^2.7.1"
timm = "^1.0.3"
pandas = "^2.2.2"
vllm = "^0.4.2"
```

{{< githubcard repo="vllm-project/vllm" >}}
<br>

## 当前 vLLM 不支持 tool call

因为当前 vLLM 不支持使用 OpenAI 包的方式传入 `tools`，详见 vLLM PR {{< ionicons "logo-github" >}}[#3237](https://github.com/vllm-project/vllm/pull/3237)。

```py
from openai import OpenAI

client = OpenAI()

messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
tools = [...]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
response_message = response.choices[0].message
tool_calls = response_message.tool_calls
```

我们需要使用对应模型的 Prompt, 传入 `messages` 中，然后通过 OpenAI 的 SDK 调用。

## ChatGLM3-6B

启动命令

```bash
python -m vllm.entrypoints.openai.api_server --served-model-name ChatGLM3-6B --model /THUDM/chatglm3-6b \
        --max-model-len 8192 --host {{HOST}} --chat-template /chat-template/chatglm3-6b-template.jinja \
        --tokenizer /THUDM/chatglm3-6b --tensor-parallel-size 4 --trust-remote-code
```

- 替换参数 `{{HOST}}` 为你需要部署模型的 IP 地址。
- 参数 `--tensor-parallel-size` 用来指定使用多少张 GPU(或 TPU)并行推理

其中 `chatglm3-6b-template.jinja` 文件内容如下:

```jinja
{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}
```

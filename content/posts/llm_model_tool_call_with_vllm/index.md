---
author: ["Simon Wei"]
title: "LLM Funcation Calling with vLLM"
date: "2024-06-05"
description: "LLM Funcation Calling with vLLM"
summary: "LLM Funcation Calling with vLLM"
tags: ["LLLM", "vLLM", "Funcation Calling"]
categories: ["LLLM", "vLLM", "Funcation Calling"]
series: ["LLM"]
ShowToc: true
TocOpen: true
math: true
---

<!-- {{< githubcard repo="vllm-project/vllm" >}} -->

## Python ENV dependency

Run `pip install poetry==1.8.0` to install [Poetry](https://python-poetry.org/), which is Python packaging and dependency management tool.

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

### vLLM(<=0.4.2) not support tool_call

This codes is the best way for tool_call.

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

Due to latest vLLM does not support using tool_call with OpenAI python SDK, releated PR {{< ionicons "logo-github" >}}[#3237](https://github.com/vllm-project/vllm/pull/3237).
We will use corresponding model **Prompt**, then insert `tools` into `request.messages`. After that, we could invoke LLM chat-completions interface with OpenAI python SDK.

### ChatGLM3-6B

Start LLM server with vLLM scripts, which will offers standard OpenAI API.

```bash
python -m vllm.entrypoints.openai.api_server --served-model-name ChatGLM3-6B --model /THUDM/chatglm3-6b \
        --max-model-len 8192 --host {{HOST}} --chat-template /chat-template/chatglm3-6b-template.jinja \
        --tokenizer /THUDM/chatglm3-6b --tensor-parallel-size 4 --trust-remote-code
```

- replace `{{HOST}}` with your target machine IP.
- `--tensor-parallel-size`: use how many GPUs/TPUs for distributed parallel inference.

`chatglm3-6b-template.jinja`:

```jinja
{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}
```

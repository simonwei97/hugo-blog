---
author: ["Simon Wei"]
title: "使用vLLM部署LLM进行Funcation Calling"
date: "2024-06-05"
description: ""
summary: ""
tags: ["LLM", "vLLM", "Tool Call"]
categories: ["LLLM", "vLLM", "Tool Call"]
series: ["LLM"]
ShowToc: true
TocOpen: true
math: true
---

{{< githubcard repo="vllm-project/vllm" >}}

# 环境依赖

运行 `pip install poetry==1.8.0` 安装 poetry 包管理。

如下为 poetry 包管理中具体的依赖包

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

## 智谱 GLM

### 启动 OpenAI-API 的 Server

```bash
# ChatGLM3-6B
python -m vllm.entrypoints.openai.api_server --served-model-name ChatGLM3-6B --model /THUDM/chatglm3-6b \
        --max-model-len 8192 --chat-template chatglm-template.jinja \
        --tokenizer /THUDM/chatglm3-6b --tensor-parallel-size 4 --trust-remote-code

# GLM-4-9B-Chat
python -m vllm.entrypoints.openai.api_server --served-model-name GLM-4-9B --model /THUDM/glm-4-9b-chat \
        --max-model-len 8192 --chat-template chatglm-template.jinja \
        --tokenizer /THUDM/glm-4-9b-chat --tensor-parallel-size 4 --trust-remote-code
```

- 参数 `--host` 指定需要部署模型的 IP 地址
- 参数 `--port` 指定需要部署模型的端口
- 参数 `--tensor-parallel-size` 用来指定使用多少张 GPU(或 TPU)并行推理
- 参数 `--max-model-len` 表示模型上下文长度，示例中指定为 8K。

{{< notice tip >}}
此处最好显式指定 `--max-model-len`，否则会使用默认值 **32768** ，导致当前机器的显存不够用，出现 OOM。
{{< /notice >}}

其中 `chatglm-template.jinja` 文件内容如下:

```jinja
{% for message in messages %}
{% if loop.first %}
[gMASK]sop<|{{ message['role'] }}|>\n{{ message['content'] }}
{% else %}
<|{{ message['role'] }}|>\n {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant|>
{% endif %}
```

### Client

Client 端可以直接使用 OpenAI 的 SDK 访问接口。

```py
from openai import OpenAI

SYTEMT_PROMPT = "Answer the following questions as best as you can. You have access to the following tools:\n{tools}"

base_url = "http://127.0.0.1:8000/v1"
client = OpenAI(api_key="EMPTY", base_url=base_url)

models = client.models.list()
model_name = models.data[0].id

query = "What's the Celsius temperature in San Francisco?"
tools = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        }
    },
]
messages = [
    {"role": "system", "content": SYTEMT_PROMPT.format(tools=tools)},
    {"role": "user", "content": query},
]

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    max_tokens=1024,
    temperature=0.8,
    presence_penalty=1.2,
    top_p=0.7,
    extra_body={
        "top_k": 40,
    }
)

resp_conent = response.choices[0].message.content
```

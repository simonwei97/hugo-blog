---
author: ["Simon Wei"]
title: "LLM tool call with vLLM"
date: "2024-06-05"
description: "LLM tool call with vLLM"
summary: "LLM tool call with vLLM"
tags: ["LLLM", "vLLM", "Tool Call"]
categories: ["LLLM", "vLLM", "Tool Call"]
series: ["LLM"]
ShowToc: true
TocOpen: true
math: true
---


# Python ENV dependency libs

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
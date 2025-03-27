# InstructLab SDG Examples with Llama Stack

This directory contains a few examples of how to get InstructLab
Synthetic Data Generation pipelines running via Llama Stack. Note that
this uses temporary forks of Llama Stack, as this is only a prototype
and may or may not end up actually becoming something worth attempting
to push upstream to Llama Stack itself.

## Prerequisites

Ensure you have a vLLM running the `meta-llama/Llama-3.2-3B-Instruct`
model. Ben uses a command like below on his Nvidia RTX4080 machine:

```
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --guided-decoding-backend outlines \
  --max-model-len 48000
```

## Install temporary forks of Llama Stack, OpenAI Python Client

The prototype code lives in @bbrowning's forks of `llama-stack`,
`llama-stack-client-python`, and `llama-stackopenai-client`
repositories, so clone those and install from source.

```
git clone --branch sdg-again https://github.com/bbrowning/llama-stack.git
git clone --branch sdg-again https://github.com/bbrowning/llama-stack-client-python.git
git clone https://github.com/bbrowning/llama-stack-openai-client.git
cd llama-stack
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ../llama-stack-client-python
pip install -e ../llama-stack-openai-client
llama stack build --template remote-vllm --image-type venv
```

## Clone and install Research SDG Repo

```
pip install git+https://github.com/Red-Hat-AI-Innovation-Team/SDG-Research
```

## Run a simple InstructLab SDG annotation pipeline


```
VLLM_URL="http://localhost:8000/v1" \
INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" \
python sdg_examples/annotation_pipeline.py
```

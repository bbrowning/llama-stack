# InstructLab SDG Examples with Llama Stack

This directory contains a few examples of how to get InstructLab
Synthetic Data Generation pipelines running via Llama Stack. Note that
this uses temporary forks of Llama Stack, as this is only a prototype
and may or may not end up actually becoming something worth attempting
to push upstream to Llama Stack itself.

## Prerequisites

Ensure you have `git` and `ollama` installed. We don't really use
ollama, but we do use its stack due to some issues gettting the
remote-vllm stack up and running.

## Install temporary forks of Llama Stack

The prototype code lives in @bbrowning's forks of `llama-stack` and
`llama-stack-client-python` repositories, so clone those and install both from source.

```
git clone --branch minimal-sdg https://github.com/bbrowning/llama-stack.git
git clone --branch minimal-sdg https://github.com/bbrowning/llama-stack-client-python.git
cd llama-stack
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ../llama-stack-client-python
llama stack build --template ollama --image-type venv
```

## Run ollama

Ideally this wouldn't be needed as we don't use it, but just do it for
now...

Open a new terminal window and run ollama:
```
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
```

## Export needed environment variables

The values below are what I used in my particular EC2 instance where I
run vLLM, but your environment may have different values. If running
the `full` pipeline example, ensure you have a Mixtral model running.

```
export OPENAI_ENDPOINT="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
export OPENAI_MODEL_ID="/home/ec2-user/.cache/instructlab/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
```

## Run a simple InstructLab SDG custom pipeline


```
python sdg_examples/custom_pipeline.py
```

## Run the end-to-end InstructLab "full" SDG pipeline against a taxonomy

This takes MUCH longer, as we do real data generation and hit a real
teacher model endpoint, specified by your OPENAI_* env variables above.

Also, we need a teacher model on disk to be able to load its
tokenizer. There are various ways to do this, such as using `ilab
model download` from InstructLab or the HuggingFace CLI.

I tend to just use the tokenizer from InstructLab's quantized
merlinite instead of downloading the entire mixtral teacher
model. Either way, export an environment variable with the path to
your teacher model:

```
export TEACHER_MODEL_PATH="~/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

```
python sdg_examples/full_pipeline.py
```

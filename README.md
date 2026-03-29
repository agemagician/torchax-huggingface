# TorchAX + HuggingFace: Run Any HuggingFace Model on TPUs

Run PyTorch HuggingFace models on Google TPUs (and GPUs) using [torchax](https://github.com/google/torchax) — no JAX rewrite needed.

[![Open Full Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_huggingface_tutorial.ipynb)
[![Open Quick Start In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_quickstart.ipynb)

## Quick Start

```python
import torchax
torchax.enable_globally()

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype="bfloat16")
model.to("jax")  # Now running on JAX

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
inputs = tokenizer("Hello, world!", return_tensors="pt")
output = model(inputs["input_ids"].to("jax"))
```

## What You'll Learn

- **Forward pass on JAX** — move any PyTorch model to JAX with `model.to("jax")`
- **JIT compilation** — 10-100x speedup via `jax.jit` and XLA
- **Text classification** — use an LLM as a classifier with prompt engineering
- **Text generation** — autoregressive decoding with StaticCache for JIT compatibility
- **Distributed inference** — tensor parallelism across multiple TPUs/GPUs
- **Mini chatbot** — end-to-end chat using Gemma's instruction template

## Repository Structure

```
torchax-huggingface/
├── README.md                                    # This file
├── blog/
│   └── torchax-huggingface-tutorial.md          # dev.to blog post
├── notebooks/
│   ├── torchax_huggingface_tutorial.ipynb        # Full tutorial (self-contained)
│   └── torchax_quickstart.ipynb                  # Quick start (~10 cells)
├── assets/
│   ├── README.md                                 # Diagram rendering instructions
│   └── diagrams/                                 # Mermaid diagram sources
│       ├── torchax-architecture.md
│       ├── jit-compilation-flow.md
│       ├── tensor-parallelism.md
│       └── ecosystem-comparison.md
└── LICENSE
```

## Prerequisites

- Python 3.10+
- A Google Colab account (free tier works) or a local machine with GPU/TPU

## Local Setup

```bash
# Install PyTorch CPU (torchax handles the accelerator via JAX)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install JAX for your accelerator
pip install -U jax[tpu]      # Google Cloud TPU
# pip install -U jax[cuda12]   # NVIDIA GPU
# pip install -U jax           # CPU only

# Install torchax and transformers
pip install torchax transformers
```

## Models

| Model | Parameters | Free Colab | Notes |
|---|---|---|---|
| `google/gemma-3-1b-it` | 1B | Yes (TPU/T4) | Default in tutorials |
| `google/gemma-3-7b-it` | 7B | Colab Pro recommended | Swap in for higher quality |
| `gpt2` | 124M | Yes | Fast, good for testing |

## Blog Post

Read the full tutorial: [Run Any HuggingFace Model on TPUs: A Beginner's Guide to TorchAX](https://dev.to/ahmed_elnaggar/run-any-huggingface-model-on-tpus-a-beginners-guide-to-torchax)

## Credits & Attribution

This project builds on the work of many people and teams:

- **[Han Qi (@qihqi)](https://github.com/qihqi)** — Author of torchax and the [original 3-part tutorial series](https://huggingface.co/blog/qihqi/huggingface-jax-01) on running HuggingFace models with JAX
  - [Part 1: Forward Pass + JIT](https://huggingface.co/blog/qihqi/huggingface-jax-01)
  - [Part 2: Distributed Inference](https://huggingface.co/blog/qihqi/huggingface-jax-02)
  - [Part 3: Autoregressive Decoding](https://huggingface.co/blog/qihqi/huggingface-jax-03)
  - [Source code](https://github.com/qihqi/learning_machine/tree/main/jax-huggingface)
- **[torchax team at Google](https://github.com/google/torchax)** — Library development and documentation
- **[HuggingFace](https://huggingface.co/)** — The transformers ecosystem
- **[JAX team at Google](https://github.com/jax-ml/jax)** — JAX, XLA compiler, and TPU support
- **[Lightning AI](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/tp.html)** — Tensor parallelism documentation

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

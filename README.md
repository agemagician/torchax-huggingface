# TorchAX + HuggingFace: Run & Fine-Tune Any HuggingFace Model on TPUs

Run and fine-tune PyTorch HuggingFace models on Google TPUs (and GPUs) using [torchax](https://github.com/google/torchax) — no JAX rewrite needed.

### Inference

[![Open Full Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_huggingface_tutorial.ipynb)
[![Open Quick Start In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_quickstart.ipynb)

### Training

[![Open Training Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_training_tutorial.ipynb)
[![Open Training Quick Start In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_training_quickstart.ipynb)

## Quick Start: Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype="bfloat16")

import torchax
torchax.enable_globally()  # Enable AFTER loading the model
model.to("jax")  # Now running on JAX

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
inputs = tokenizer("Hello, world!", return_tensors="pt")
output = model(inputs["input_ids"].to("jax"))
```

## Quick Start: Training with LoRA

```python
import torchax as tx
import torchax.train

# Set up functional training (params separated from frozen buffers)
step_fn = tx.train.make_train_step(model_fn, loss_fn, optimizer)

# Training loop — one line per step
for batch in dataloader:
    loss, params, opt_state = step_fn(params, buffers, opt_state, batch, batch["labels"])
```

## What You'll Learn

### Inference (Part 1)
- **Forward pass on JAX** — move any PyTorch model to JAX with `model.to("jax")`
- **JIT compilation** — 10-100x speedup via `jax.jit` and XLA
- **Text classification** — use an LLM as a classifier with prompt engineering
- **Text generation** — autoregressive decoding with StaticCache for JIT compatibility
- **Distributed inference** — tensor parallelism across multiple TPUs/GPUs
- **Mini chatbot** — end-to-end chat using Gemma's instruction template

### Training (Part 2)
- **LoRA fine-tuning** — parameter-efficient training with PEFT (0.22% of params)
- **Full fine-tuning** — train all parameters with memory-efficient AdaFactor
- **Functional training** — `make_train_step` + optax optimizers for JIT-compiled training
- **Evaluation** — before/after loss, perplexity, and qualitative comparison
- **Save & reload** — save fine-tuned model and reload for inference

## Repository Structure

```
torchax-huggingface/
├── README.md                                    # This file
├── blog/
│   ├── torchax-huggingface-tutorial.md          # Part 1: Inference blog post
│   └── torchax-training-tutorial.md             # Part 2: Training blog post
├── notebooks/
│   ├── torchax_huggingface_tutorial.ipynb        # Full inference tutorial
│   ├── torchax_quickstart.ipynb                  # Inference quick start (~10 cells)
│   ├── torchax_training_tutorial.ipynb           # Full training tutorial
│   └── torchax_training_quickstart.ipynb         # Training quick start (~10 cells)
├── assets/
│   ├── README.md                                 # Diagram rendering instructions
│   └── diagrams/                                 # Mermaid diagram sources
│       ├── torchax-architecture.md               # Core architecture
│       ├── jit-compilation-flow.md               # JIT compilation pipeline
│       ├── tensor-parallelism.md                 # Multi-device parallelism
│       ├── ecosystem-comparison.md               # Framework comparison
│       ├── training-loop-flow.md                 # Training pipeline flow
│       ├── lora-architecture.md                  # LoRA weight decomposition
│       └── training-vs-inference.md              # Training vs inference paths
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

# Install torchax, transformers, and training dependencies
pip install -U torchax transformers flax peft datasets optax
```

## Models

| Model | Parameters | Free Colab | Notes |
|---|---|---|---|
| `google/gemma-4-E2B-it` | 2B | Yes (TPU) | Default in training tutorials |
| `google/gemma-3-1b-it` | 1B | Yes (TPU/T4) | Default in inference tutorials |
| `google/gemma-3-7b-it` | 7B | Colab Pro recommended | Swap in for higher quality |
| `gpt2` | 124M | Yes | Fast, good for testing |

## Blog Posts

1. [Run Any HuggingFace Model on TPUs: A Beginner's Guide to TorchAX](https://dev.to/ahmed_elnaggar/run-any-huggingface-model-on-tpus-a-beginners-guide-to-torchax) — Inference
2. [Fine-Tune Any HuggingFace Model on TPUs with TorchAX](https://dev.to/ahmed_elnaggar/fine-tune-any-huggingface-model-on-tpus-with-torchax) — Training

## Credits & Attribution

This project builds on the work of many people and teams:

- **[Han Qi (@qihqi)](https://github.com/qihqi)** — Author of torchax, the [PEFT training example](https://github.com/google/torchax/blob/main/examples/peft_lora_training.py), and the [original 3-part tutorial series](https://huggingface.co/blog/qihqi/huggingface-jax-01) on running HuggingFace models with JAX
  - [Part 1: Forward Pass + JIT](https://huggingface.co/blog/qihqi/huggingface-jax-01)
  - [Part 2: Distributed Inference](https://huggingface.co/blog/qihqi/huggingface-jax-02)
  - [Part 3: Autoregressive Decoding](https://huggingface.co/blog/qihqi/huggingface-jax-03)
  - [Source code](https://github.com/qihqi/learning_machine/tree/main/jax-huggingface)
- **[torchax team at Google](https://github.com/google/torchax)** — Library development and documentation
- **[HuggingFace](https://huggingface.co/)** — The transformers, PEFT, and datasets ecosystem
- **[Databricks](https://www.databricks.com/)** — Dolly 15k dataset used in training tutorials
- **[JAX team at Google](https://github.com/jax-ml/jax)** — JAX, XLA compiler, and TPU support
- **[Lightning AI](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/tp.html)** — Tensor parallelism documentation

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

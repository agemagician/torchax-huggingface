---
title: "Fine-Tune Any HuggingFace Model on TPUs with TorchAX"
published: false
description: "Learn how to fine-tune PyTorch HuggingFace models on Google TPUs using torchax and LoRA — no JAX rewrite needed. Includes evaluation, save/reload, and a Colab notebook."
tags: machinelearning, pytorch, python, tutorial
cover_image: https://raw.githubusercontent.com/ahmed-elnaggar/torchax-huggingface/main/assets/cover-image.png
series: "TorchAX + HuggingFace"
---

## What if you could fine-tune any HuggingFace model on TPUs — using PyTorch code?

Here is what the end result looks like:

```python
import torchax as tx
import torchax.train

# One function: forward → loss → gradients → optimizer update
step_fn = tx.train.make_train_step(model_fn, loss_fn, optimizer)

# Training loop
for batch in dataloader:
    loss, params, opt_state = step_fn(params, buffers, opt_state, batch, batch["labels"])
```

Your PyTorch model. JAX's training primitives. Running on TPU. No rewrite needed.

In the [first part of this series](https://dev.to/ahmed_elnaggar/run-any-huggingface-model-on-tpus-a-beginners-guide-to-torchax), we ran HuggingFace models on JAX for fast inference. Now we take the next step: **training**. We will instruction-tune Gemma 4 E2B on the Databricks Dolly 15k dataset using LoRA and torchax's functional training API — all on a free Colab TPU.

[![Open Full Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_training_tutorial.ipynb) [![Open Quick Start In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_training_quickstart.ipynb)

---

## Why Train on TPUs?

Google's Tensor Processing Units (TPUs) are purpose-built for matrix operations — the bread and butter of deep learning. Free Colab gives you access to a TPU v2-8 with ~15GB of high-bandwidth memory. That is enough to fine-tune a 2B parameter model with LoRA.

But training on TPUs traditionally meant rewriting your model in JAX (Flax, Equinox) or using PyTorch/XLA. **torchax** offers a third path: keep your PyTorch model, but use JAX's functional training primitives.

### How torchax Training Differs from Standard PyTorch

| Standard PyTorch | torchax |
|---|---|
| `loss.backward()` | `jax.value_and_grad(loss_fn)(params, ...)` |
| `optimizer.step()` | `optax.apply_updates(params, updates)` |
| Model holds its own state | Params and buffers are separate pytrees |
| Eager execution | JIT-compiled training steps |

The key difference: **functional training**. Instead of calling `loss.backward()` and `optimizer.step()` on a stateful model, torchax separates the model into immutable weight pytrees and passes them through pure functions. This is what enables JAX's `jax.jit` to compile the entire training step into a single optimized program.

---

## Prerequisites & Setup

**What you need:**
- Python 3.10+
- Basic familiarity with PyTorch and HuggingFace transformers
- A Google Colab account (free tier works with LoRA)

**Zero-setup option:** Click the Colab badge above. The notebook handles all installation automatically.

**Local setup:**

```bash
# PyTorch CPU (torchax handles the accelerator via JAX)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# JAX for your accelerator
pip install -U jax[tpu]      # Google Cloud TPU
# pip install -U jax[cuda12]   # NVIDIA GPU

# torchax, transformers, and training dependencies
pip install -U torchax transformers flax peft datasets optax
```

---

## Key Concepts for Training

Before writing code, let's understand the four concepts that make torchax training work.

### 1. Param/Buffer Separation

JAX's `jax.value_and_grad` needs to know *which* inputs to differentiate. In standard PyTorch, the model owns its weights. In torchax training, we explicitly separate:

- **params** — trainable parameters (get gradients)
- **buffers** — everything else (frozen weights, running stats, constants)

```python
params = {n: p for n, p in model.named_parameters() if p.requires_grad}
frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
buffers = dict(model.named_buffers())
buffers.update(frozen)
```

For LoRA, `params` contains only the tiny adapter weights (~0.5% of the model). For full fine-tuning, it contains everything.

### 2. optax Optimizers

Unlike PyTorch optimizers (which carry hidden mutable state), optax optimizers are **pure functions**:

```python
# PyTorch: hidden state inside optimizer
optimizer.step()

# optax: explicit state, no hidden pockets
updates, new_opt_state = optimizer.update(grads, opt_state, params)
new_params = optax.apply_updates(params, updates)
```

This functional design means the optimizer state is just another pytree that flows through the training step — perfect for `jax.jit`.

### 3. make_train_step

`torchax.train.make_train_step()` is the central API. It composes three pieces into a single JIT-compilable function:

1. **model_fn** — a pure function: `(weights, buffers, batch) → output`
2. **loss_fn** — extracts the scalar loss: `(output, labels) → loss`
3. **optimizer** — an optax optimizer

The result is `step_fn(params, buffers, opt_state, batch, labels) → (loss, new_params, new_opt_state)`.

Under the hood, this uses `jax.value_and_grad` for efficient gradient computation and `optax.apply_updates` for weight updates — all compiled into a single XLA program.

### 4. Full Fine-Tuning vs LoRA

| | Full Fine-Tuning | LoRA |
|---|---|---|
| **Trainable params** | All (~2B) | Tiny adapters (~0.5%) |
| **Memory** | ~18-20 GB | ~5-7 GB |
| **Speed** | Slower | Faster |
| **Quality** | Higher ceiling | Nearly as good |
| **Free Colab TPU** | Tight / may OOM | Fits comfortably |

**LoRA** (Low-Rank Adaptation) freezes the base model and adds small trainable matrices to attention layers. Instead of updating the full weight matrix W, it learns a low-rank decomposition: `W + (α/r) × B·A` where A and B are tiny matrices.

For free Colab, LoRA is the recommended path.

---

## Step 1: Load and Prepare the Dataset

We use [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) — 15,000 human-written instruction-response pairs across 7 categories (QA, summarization, brainstorming, etc.).

```python
import datasets as hf_datasets
from transformers import AutoTokenizer

MODEL_NAME = "google/gemma-4-E2B-it"
DATASET_NAME = "databricks/databricks-dolly-15k"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

raw_dataset = hf_datasets.load_dataset(DATASET_NAME, split="train")
```

Each example has an `instruction`, optional `context`, `response`, and `category`. We format these into Gemma's chat template:

```python
def format_example(example):
    user_content = example["instruction"]
    if example.get("context", ""):
        user_content += f"\n\nContext: {example['context']}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}
```

Then tokenize and create dataloaders:

```python
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

# Subset, split, tokenize
subset = raw_dataset.shuffle(seed=42).select(range(2200))
split = subset.train_test_split(test_size=200, seed=42)

def tokenize_example(example):
    formatted = format_example(example)
    return tokenizer(formatted["text"], padding="max_length", max_length=512, truncation=True)

train_tokenized = split["train"].map(tokenize_example, remove_columns=split["train"].column_names)
eval_tokenized = split["test"].map(tokenize_example, remove_columns=split["test"].column_names)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(train_tokenized, shuffle=True, collate_fn=collator, batch_size=2)
eval_dataloader = DataLoader(eval_tokenized, shuffle=False, collate_fn=collator, batch_size=2)
```

---

## Step 2: Load the Model and Apply LoRA

Here is where the torchax pattern matters: load the model with torchax **disabled**, then enable it before moving to JAX.

```python
import torch
import torchax as tx
import peft

# Load model with torchax disabled to avoid intercepting init ops
with tx.disable_temporarily():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    )
```

**Why disable?** HuggingFace model initialization uses operations (like in-place tensor filling) that torchax does not support. Disabling torchax during loading keeps everything on CPU, then we move to JAX after.

Now apply LoRA:

```python
peft_config = peft.LoraConfig(
    task_type=peft.TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                            # Rank of the LoRA matrices
    lora_alpha=32,                   # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Which attention layers to adapt
)
model = peft.get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: trainable params: 5,767,168 || all params: 2,619,206,656 || trainable%: 0.22%
```

Only 0.22% of parameters are trainable — that is the power of LoRA.

Finally, enable torchax and move to the JAX device:

```python
tx.enable_globally()
device = torch.device("jax")
model.to(device)
model.train()
```

---

## Step 3: Baseline Evaluation

Before training, we measure the model's performance to compare against later:

```python
import math

def evaluate_loss(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1
    model.train()
    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss, math.exp(min(avg_loss, 100))

baseline_loss, baseline_ppl = evaluate_loss(model, eval_dataloader, device)
print(f"Baseline loss: {baseline_loss:.4f}, perplexity: {baseline_ppl:.2f}")
```

We also generate sample responses for qualitative comparison after training.

---

## Step 4: Set Up Functional Training

This is where torchax diverges from standard PyTorch. We separate the model, create an optax optimizer, and compose everything into a JIT-compiled training step.

### Separate params and buffers

```python
import optax
import torchax.train

params = {n: p for n, p in model.named_parameters() if p.requires_grad}
buffers = dict(model.named_buffers())
frozen_params = {n: p for n, p in model.named_parameters() if not p.requires_grad}
buffers.update(frozen_params)
```

### Create the optimizer

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=2e-4, warmup_steps=50, decay_steps=500
)
optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
opt_state = tx.interop.call_jax(optimizer.init, params)
```

Note `tx.interop.call_jax` — this bridges optax's JAX calls with torchax tensors.

### Define model_fn and loss_fn

```python
def model_fn(weights, buffers, batch):
    """Stateless forward pass using functional_call."""
    return torch.func.functional_call(
        model, {**weights, **buffers}, args=(), kwargs=batch
    )

def loss_fn(model_output, labels):
    """Extract loss from HuggingFace model output."""
    return model_output.loss
```

`torch.func.functional_call` runs the model as a pure function — no hidden state, just inputs and outputs. This is what enables JAX to trace and compile it.

### Compose into a training step

```python
step_fn = tx.train.make_train_step(model_fn, loss_fn, optimizer)
```

That single line creates a function that does: forward pass → loss computation → gradient calculation → optimizer update — all compiled into one XLA program.

---

## Step 5: The Training Loop

```python
import time
from tqdm.auto import tqdm

torch.manual_seed(42)
train_losses = []
start_time = time.time()

for epoch in range(1):
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for step, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        loss, params, opt_state = step_fn(
            params, buffers, opt_state, batch, batch["labels"]
        )

        train_losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

elapsed = time.time() - start_time
print(f"Training complete! {len(train_losses)} steps in {elapsed:.0f}s")
```

**What to expect:**
- **Step 1:** ~30-60 seconds (JAX compiles the entire training step)
- **Steps 2+:** ~1-3 seconds each (running the compiled program)
- **Total:** ~20-40 minutes for 2000 samples with LoRA on free Colab TPU

The first step is slow because JAX traces through the entire model, loss computation, gradient calculation, and optimizer update — then compiles it all into a single optimized XLA program. Every subsequent step reuses this compiled program.

---

## Step 6: Evaluate the Improvement

After training, we compare against our baseline:

```python
# Load trained params back into model
with torch.no_grad():
    for name, param in params.items():
        parts = name.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], torch.nn.Parameter(param))

final_loss, final_ppl = evaluate_loss(model, eval_dataloader, device)

print(f"{'Metric':<20} {'Before':>10} {'After':>10}")
print(f"{'Loss':<20} {baseline_loss:>10.4f} {final_loss:>10.4f}")
print(f"{'Perplexity':<20} {baseline_ppl:>10.2f} {final_ppl:>10.2f}")
```

You should see loss decrease and perplexity improve after training. The qualitative comparison (generated responses before vs. after) is even more telling — the fine-tuned model produces more focused, instruction-following responses.

---

## Step 7: Save and Reload

### Save

Convert JAX arrays back to CPU tensors and save using HuggingFace's standard format:

```python
import numpy as np

save_dir = "./fine_tuned_model"

with torch.no_grad():
    cpu_state_dict = {
        name: torch.tensor(np.array(p)).contiguous()
        for name, p in params.items()
    }
    model.save_pretrained(save_dir, state_dict=cpu_state_dict)

tokenizer.save_pretrained(save_dir)
```

For LoRA, this saves only the tiny adapter weights (~20MB). For full fine-tuning, it saves the entire model (~4GB).

### Reload

```python
with tx.disable_temporarily():
    # For LoRA: load base model + adapters separately
    reloaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    )
    reloaded_model = peft.PeftModel.from_pretrained(reloaded_model, save_dir)

reloaded_model.to(device)
reloaded_model.eval()
```

The pattern is the same as loading: disable torchax, load on CPU, then move to JAX. For LoRA models, you load the base model first, then attach the saved adapters with `PeftModel.from_pretrained()`.

---

## Full Fine-Tuning: When LoRA Is Not Enough

The notebook supports full fine-tuning by changing one setting:

```python
TRAINING_MODE = "full"
```

This trains all 2B parameters instead of just the LoRA adapters. The trade-off is much higher memory usage. To make it fit on free Colab TPU:

- **AdaFactor optimizer** — uses ~50% less memory than AdamW (stores only row/column statistics instead of per-parameter moments)
- **Reduced sequence length** — `MAX_SEQ_LEN = 256` halves activation memory
- **Smaller batch size** — `BATCH_SIZE = 1` with higher gradient accumulation steps

```python
USE_ADAFACTOR = True
USE_GRADIENT_CHECKPOINTING = True

if TRAINING_MODE == "full" and USE_ADAFACTOR:
    optimizer = optax.adafactor(learning_rate=schedule)
else:
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
```

Full fine-tuning gives a higher quality ceiling but LoRA gets you 90%+ of the way with a fraction of the compute.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `OutOfMemoryError` | Model + optimizer too large | Switch to LoRA, reduce `BATCH_SIZE` or `MAX_SEQ_LEN` |
| `TypeError: not a valid JAX type` | Custom HuggingFace type not registered | Register with `jax.tree_util.register_pytree_node()` |
| `Loss is NaN` | Learning rate too high | Reduce by 10x, ensure `torch_dtype=torch.bfloat16` |
| `Slow first step` | Normal — JAX JIT compilation | Wait ~30-60s; subsequent steps are fast |
| `make_train_step error` | API mismatch | Update: `pip install -U torchax` |

---

## The Big Picture: Inference + Training

With the [inference tutorial](https://dev.to/ahmed_elnaggar/run-any-huggingface-model-on-tpus-a-beginners-guide-to-torchax) and this training tutorial, you now have the complete torchax story:

1. **Run** any HuggingFace model on TPU (`model.to("jax")`)
2. **Benchmark** with JIT compilation (10-100x speedup)
3. **Fine-tune** with LoRA or full training (`make_train_step`)
4. **Save** and reload for production inference

All using PyTorch code. No JAX rewrite needed.

---

## Resources

- **Notebooks:**
  - [Full training tutorial](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_training_tutorial.ipynb) — all the code from this post, ready to run
  - [Training quickstart](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_training_quickstart.ipynb) — same pipeline in ~10 cells
  - [Inference tutorial](https://colab.research.google.com/github/ahmed-elnaggar/torchax-huggingface/blob/main/notebooks/torchax_huggingface_tutorial.ipynb) — Part 1 of this series
- **Libraries:**
  - [torchax GitHub](https://github.com/google/torchax)
  - [PEFT/LoRA documentation](https://huggingface.co/docs/peft)
  - [optax documentation](https://optax.readthedocs.io/)
- **References:**
  - [torchax PEFT LoRA example](https://github.com/google/torchax/blob/main/examples/peft_lora_training.py) — the official example this tutorial builds on
  - [Han Qi's tutorial series](https://huggingface.co/blog/qihqi/huggingface-jax-01) — the original 3-part series on torchax + HuggingFace

## Credits

- **[Han Qi (@qihqi)](https://github.com/qihqi)** — author of torchax, PEFT training example, and the original tutorial series
- **[torchax team at Google](https://github.com/google/torchax)** — library development
- **[HuggingFace](https://huggingface.co/)** — transformers, PEFT, and datasets ecosystem
- **[Databricks](https://www.databricks.com/)** — Dolly 15k dataset
- **[JAX team at Google](https://github.com/jax-ml/jax)** — JAX, XLA, and TPU support

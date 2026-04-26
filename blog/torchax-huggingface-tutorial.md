---
title: "Run Any HuggingFace Model on TPUs: A Beginner's Guide to TorchAX"
published: false
description: "Learn how to run PyTorch HuggingFace models on Google TPUs using torchax — no JAX rewrite needed. Includes benchmarks, text classification, text generation, and a Colab notebook."
tags: machinelearning, pytorch, python, tutorial
cover_image: https://raw.githubusercontent.com/agemagician/torchax-huggingface/main/assets/cover-image.png
series: "TorchAX + HuggingFace"
---

## What if you could run any HuggingFace model on TPUs — without rewriting a single line of model code?

Here is what the end result looks like:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype="bfloat16")

import torchax
torchax.enable_globally()  # Enable AFTER loading the model

model.to("jax")  # That's it. Now running on JAX.
```

Five lines. Your PyTorch model is now executing on JAX — with access to TPUs, JIT compilation, and automatic parallelism across devices.

In this tutorial, we will go from zero to building a working chatbot powered by a HuggingFace model running on JAX. Along the way, you will learn key JAX concepts, see real benchmarks, and understand *why* this approach exists.

[![Open Full Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agemagician/torchax-huggingface/blob/main/notebooks/torchax_huggingface_tutorial.ipynb) [![Open Quick Start In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agemagician/torchax-huggingface/blob/main/notebooks/torchax_quickstart.ipynb)

---

## Why This Matters: The HuggingFace + JAX Problem

In 2024, HuggingFace [removed native JAX and TensorFlow support](https://github.com/huggingface/transformers/issues/25004) from its `transformers` library to focus development on PyTorch. This left thousands of JAX users — especially those running on Google Cloud TPUs — without a straightforward way to use HuggingFace's massive model collection.

### What is JAX?

If you are new to JAX, think of it as Google's high-performance numerical computing library. It looks like NumPy on the surface, but under the hood it offers three powerful capabilities:

1. **JIT Compilation** — JAX can compile your Python code into optimized machine code using the XLA compiler. The first run is slower (compilation), but every subsequent call is dramatically faster.

2. **TPU Support** — JAX is the native programming model for Google's Tensor Processing Units. If you want to use TPUs, JAX is the most natural path.

3. **Automatic Parallelism** — JAX can automatically distribute computation across multiple devices (TPUs or GPUs) using a single-program model called gSPMD. You describe *what* should be sharded; the compiler figures out *how*.

### Enter TorchAX

[**torchax**](https://github.com/google/torchax) is a library from Google that bridges PyTorch and JAX. It works by creating a special `torch.Tensor` subclass that secretly holds a `jax.Array` inside. When PyTorch operations are called on this tensor, torchax intercepts them and executes the JAX equivalent instead.

Think of it like a Trojan horse: PyTorch *thinks* it is working with regular tensors, but the computation is actually happening on JAX.

```
PyTorch Model
    |
    v
torchax.Tensor (looks like torch.Tensor)
    |
    v
jax.Array (actual computation on TPU/GPU)
```

This means you can take *any* PyTorch model — including HuggingFace models — and run it on JAX without modifying the model code at all.

> **Credits:** This tutorial builds on the excellent [3-part blog series](https://huggingface.co/blog/qihqi/huggingface-jax-01) by Han Qi ([@qihqi](https://github.com/qihqi)), the author of torchax, and on the [torchax documentation](https://google.github.io/torchax/). We expand on those tutorials with beginner-friendly explanations, a different model (Gemma instead of Llama), benchmarks, and a complete Colab-ready notebook.

---

## TorchAX vs. the Alternatives

Before diving into code, it helps to understand where torchax fits in the broader ecosystem:

| Approach | Effort | Performance | Best For |
|---|---|---|---|
| **Rewrite in Flax/Equinox** | High (full rewrite) | Native JAX speed | New projects starting in JAX |
| **torch-xla (PyTorch/XLA)** | Low (add XLA device) | Good (XLA compiled) | PyTorch training on TPUs |
| **torchax** | Low (change device to 'jax') | Great (JAX JIT + interop) | Running HF models on JAX, mixing PyTorch + JAX |
| **ONNX export** | Medium (export + runtime) | Variable | Cross-framework deployment |

**When should you use torchax?** When you have a PyTorch model (especially from HuggingFace) and want to leverage JAX's JIT compilation, TPU support, or interop with JAX libraries — without rewriting the model.

---

## Prerequisites & Setup

**What you need:**
- Python 3.10+
- Basic familiarity with PyTorch (loading models, running inference)
- A Google Colab account (free tier works for the 1B model)

**Zero-setup option:** Click the Colab badge above. The notebook handles all installation automatically.

**Local setup:**

```bash
# 1. Install PyTorch (CPU version — torchax handles the accelerator)
pip install torch --index-url https://download.pytorch.org/whl/cpu  # Linux
# pip install torch  # macOS

# 2. Install JAX for your accelerator
pip install -U jax[tpu]     # Google Cloud TPU
# pip install -U jax[cuda12]  # NVIDIA GPU
# pip install -U jax          # CPU only

# 3. Install torchax, transformers, and flax (for JAX compatibility)
pip install -U torchax transformers flax
```

---

## Key Concepts for Beginners

Before we write code, let's demystify three JAX concepts you will encounter throughout this tutorial.

### Pytrees: JAX's Data Containers

A **pytree** is any nested structure of Python containers (dicts, lists, tuples) with arrays as leaves. JAX uses pytrees everywhere — model weights are pytrees, function inputs/outputs are pytrees.

Think of a pytree like a shipping box with labeled compartments. JAX knows how to open standard boxes (dicts, lists, tuples), pull out all the arrays, do math on them, and put them back.

The catch: JAX does not know how to open *custom* boxes. HuggingFace defines custom output types like `CausalLMOutputWithPast` — we need to teach JAX how to unpack and repack these. This is called **pytree registration**, and we will see it in action shortly.

### JIT Compilation: Translate Once, Run Fast Forever

**JIT** (Just-In-Time) compilation is like translating a recipe from English to machine code. The first time you call a JIT-compiled function, JAX traces through it, records all the operations, and compiles an optimized version. Subsequent calls skip the tracing and run the compiled version directly.

```
First call:  Python code → trace → compile → execute  (slow)
Second call: compiled code → execute                   (fast!)
```

The speedup can be 10-100x or more. The trade-off is that the compiled function is specialized for the input shapes it was traced with — if shapes change, JAX recompiles.

### Static vs. Dynamic Values

When JAX traces a function for JIT, it treats inputs as abstract shapes, not concrete values. If your code has a branch like `if use_cache:`, JAX cannot evaluate it during tracing because `use_cache` is abstract. This causes a `ConcretizationTypeError`.

The fix: mark such values as **static** (compile-time constants) so JAX knows their actual value during tracing. We will see two ways to do this: closures and `static_argnums`.

---

## Step 1: Your First Forward Pass

Let's load a model and run it on JAX. We will use [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it) — a small, instruction-tuned model from Google that runs comfortably on free Colab hardware.

```python
import torch
import torchax
import jax
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cpu"
)

# Enable torchax globally AFTER model loading
# This prevents intercepting unsupported initialization ops
torchax.enable_globally()

# Move model weights to the JAX device
model.to("jax")

# Tokenize an input prompt
prompt = "The secret to baking a good cake is"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to("jax")

# Run a forward pass (eager mode)
start = time.perf_counter()
with torch.no_grad():
    outputs = model(input_ids, use_cache=False)
elapsed = time.perf_counter() - start

print(f"Output logits shape: {outputs.logits.shape}")
print(f"Eager forward pass: {elapsed:.3f}s")
```

**What happened:**
1. We load the model on CPU first, *then* call `torchax.enable_globally()`. This ordering is important — enabling torchax before model loading can intercept unsupported initialization ops and cause errors.
2. `model.to("jax")` moves every parameter from CPU to the JAX device — just like `model.to("cuda")` for GPUs.
3. The forward pass runs through PyTorch's code path, but every operation is executed by JAX under the hood.

The output `logits` tensor has shape `(1, sequence_length, vocab_size)`. Each position contains a score for every token in the vocabulary — the highest score is the model's prediction for the next token.

---

## Step 2: Speed It Up with JIT Compilation

The eager forward pass works, but it is slow — every operation goes through Python one at a time. Let's compile the model for dramatically faster inference.

### The extract_jax Approach

The `torchax.extract_jax()` function converts a PyTorch model into a pure JAX function:

```python
# Extract a JAX-callable function and the model weights as a pytree
weights, jax_func = torchax.extract_jax(model)
```

This returns two things:
- `weights` — the model's state_dict as a pytree of `jax.Array`s
- `jax_func` — a function with signature `jax_func(weights, args_tuple, kwargs_dict)`

### Register HuggingFace Output Types as Pytrees

Before we can JIT this function, we need to teach JAX about HuggingFace's custom types:

```python
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs, cache_utils

# Register CausalLMOutputWithPast
def output_flatten(v):
    return v.to_tuple(), None

def output_unflatten(aux, children):
    return modeling_outputs.CausalLMOutputWithPast(*children)

register_pytree_node(
    modeling_outputs.CausalLMOutputWithPast,
    output_flatten,
    output_unflatten,
)

# Register DynamicCache
def _flatten_dynamic_cache(cache):
    return (cache.key_cache, cache.value_cache), None

def _unflatten_dynamic_cache(aux, children):
    c = cache_utils.DynamicCache()
    c.key_cache, c.value_cache = children
    return c

register_pytree_node(
    cache_utils.DynamicCache,
    _flatten_dynamic_cache,
    _unflatten_dynamic_cache,
)
```

### Handle Static Arguments with a Closure

The `use_cache` flag is a boolean that JAX cannot trace. We wrap it in a closure to make it a compile-time constant:

```python
def forward_no_cache(weights, input_ids):
    return jax_func(weights, (input_ids,), {"use_cache": False})

jitted_forward = jax.jit(forward_no_cache)
```

### Benchmark: Eager vs. JIT

```python
# Convert input to a native JAX array for jax.jit
jax_input_ids = jax.device_put(inputs["input_ids"].numpy())

# Warm up (first call triggers compilation)
res = jitted_forward(weights, jax_input_ids)
jax.block_until_ready(res)

# Benchmark 3 runs
for i in range(3):
    start = time.perf_counter()
    res = jitted_forward(weights, jax_input_ids)
    jax.block_until_ready(res)
    elapsed = time.perf_counter() - start
    print(f"Run {i}: {elapsed:.4f}s")
```

Expected output (times will vary by hardware):

```
Run 0: 0.0142s  # Already compiled from warm-up
Run 1: 0.0038s
Run 2: 0.0035s
```

The JIT-compiled version runs orders of magnitude faster than eager mode. This is the power of XLA compilation — operations are fused, memory is optimized, and the accelerator runs a single optimized program.

---

## Step 3: The Simpler API — torchax.compile

The `extract_jax` + manual JIT approach gives you full control, but for most cases there is a simpler way. The catch is that `torchax.compile()` uses `jax.jit` under the hood, so we need to avoid passing dynamic boolean flags like `use_cache`. We wrap the model in a thin module that bakes in these constants:

```python
import torch.nn as nn

class NoCacheModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids):
        # Return only logits to avoid HuggingFace output class pytree issues
        return self.base_model(input_ids, use_cache=False, return_dict=False)[0]

# One-liner: compile the wrapped model
compiled_model = torchax.compile(NoCacheModel(model))

# Use it like a normal PyTorch model
with torch.no_grad():
    logits = compiled_model(input_ids)
```

Under the hood, `torchax.compile()` wraps your model in a `JittableModule` and applies `jax.jit`. The first call triggers compilation; subsequent calls are fast. The `NoCacheModel` wrapper ensures that boolean flags are constants (not traced) and that the output is a plain tensor (not a custom HuggingFace type that needs pytree registration).

---

## Step 4: Text Classification

Let's use our JIT-compiled model for a practical task — sentiment classification. Since Gemma is an instruction-tuned model, we can use prompt engineering:

```python
def classify_sentiment(text, model, tokenizer):
    prompt = f"""Classify the following text as POSITIVE or NEGATIVE.
Text: "{text}"
Classification:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("jax")

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)

    # Get the predicted next token
    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    prediction = tokenizer.decode([next_token_id]).strip()
    return prediction

# Test it
texts = [
    "This movie was absolutely fantastic, I loved every minute!",
    "The service was terrible and the food was cold.",
    "A perfectly average experience, nothing special.",
]

for text in texts:
    result = classify_sentiment(text, model, tokenizer)
    print(f"Text: {text[:50]}...  =>  {result}")
```

---

## Step 5: Text Generation (Autoregressive Decoding)

Classification is useful, but the real power of LLMs is generating text. Let's understand how this works.

### How Autoregressive Decoding Works

An LLM predicts one token at a time. Given an input of length `n`, it produces scores for the next token. We pick one (e.g., the highest-scoring token via greedy decoding), append it to the input, and repeat:

```
Iteration 1: input (1, n)     → output (1, n)     → pick token
Iteration 2: input (1, n+1)   → output (1, n+1)   → pick token
Iteration 3: input (1, n+2)   → output (1, n+2)   → pick token
...
```

The problem: **input shapes change every iteration**. JIT compilation specializes for fixed shapes, so changing shapes means recompilation every step — worse than eager mode.

### The KV Cache Solution

The **KV (Key-Value) cache** stores intermediate computations from previous tokens so the model only needs to process the *new* token each iteration:

```
Iteration 1: input (1, n)              → output + kv_cache(n)
Iteration 2: input (1, 1) + cache(n)   → output + kv_cache(n+1)
Iteration 3: input (1, 1) + cache(n+1) → output + kv_cache(n+2)
```

With a **DynamicCache**, the cache grows each step — shapes still change. With a **StaticCache**, the cache has a fixed maximum length — shapes stay constant, making it JIT-friendly.

### Implementation with StaticCache

```python
from transformers.cache_utils import StaticCache

# Register StaticCache as a pytree
def _flatten_static_cache(cache):
    return (
        cache.key_cache, cache.value_cache
    ), (cache.config, cache.max_batch_size, cache.max_cache_len,
        getattr(cache, "device", None), getattr(cache, "dtype", None))

def _unflatten_static_cache(aux, children):
    config, max_batch_size, max_cache_len, device, dtype = aux
    kwargs = {}
    if device is not None: kwargs["device"] = device
    if dtype is not None: kwargs["dtype"] = dtype
    cache = StaticCache(config, max_batch_size, max_cache_len, **kwargs)
    cache.key_cache, cache.value_cache = children
    return cache

register_pytree_node(
    StaticCache,
    _flatten_static_cache,
    _unflatten_static_cache,
)

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("jax")
    batch_size, seq_length = input_ids.shape

    # Create a static cache with fixed maximum length
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=seq_length + max_new_tokens,
        device="jax",
        dtype=model.dtype,
    )
    cache_position = torch.arange(seq_length, device="jax")

    # Prefill: process the full prompt
    with torch.no_grad():
        logits, past_key_values = model(
            input_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        )

    next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    generated_ids = [next_token[:, 0].item()]
    cache_position = torch.tensor([seq_length], device="jax")

    # Decode: generate one token at a time
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            logits, past_key_values = model(
                next_token,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True,
            )
        next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        token_id = next_token[:, 0].item()

        if token_id == tokenizer.eos_token_id:
            break
        generated_ids.append(token_id)
        cache_position += 1

    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# Generate!
result = generate_text(model, tokenizer, "The secret to baking a good cake is")
print(result)
```

---

## Step 6: Distributed Inference (Tensor Parallelism)

If you have access to multiple devices (e.g., a TPU v2-8 with 8 chips, or multi-GPU), you can shard the model weights across devices for faster inference.

### How Tensor Parallelism Works

In tensor parallelism, we split weight matrices across devices:
- **Column-parallel**: Q, K, V, Gate, and Up projections are split along the output dimension
- **Row-parallel**: O and Down projections are split along the input dimension
- Between these two, only a single **all-reduce** operation is needed per layer

JAX's gSPMD handles the communication automatically — you just specify how each weight should be sharded.

### Sharding the Weights

```python
from jax.sharding import PartitionSpec as P, NamedSharding

# Create a device mesh
mesh = jax.make_mesh((jax.device_count(),), ("axis",))

def shard_weights(mesh, weights):
    sharded = {}
    for name, tensor in weights.items():
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
            spec = P("axis", None)  # Column-parallel
        elif any(k in name for k in ["o_proj", "down_proj", "lm_head", "embed_tokens"]):
            spec = P(None, "axis")  # Row-parallel
        else:
            spec = P()  # Replicate (e.g., layer norms)
        sharded[name] = jax.device_put(tensor, NamedSharding(mesh, spec))
    return sharded

# Apply sharding
weights, jax_func = torchax.extract_jax(model)
weights = shard_weights(mesh, weights)

# Replicate the input across all devices
input_ids_sharded = jax.device_put(
    inputs["input_ids"], NamedSharding(mesh, P())
)
```

With sharded weights, the same `jax.jit`-compiled function now runs in parallel across all devices. The XLA compiler automatically inserts the necessary all-reduce operations.

> **Note:** Tensor parallelism requires a multi-device environment. On free Colab TPU (single device), this section is for illustration. Use a TPU v2-8 or multi-GPU setup to run it.

---

## Step 7: Build a Mini Chatbot

Let's wrap everything into a simple chat function using Gemma's instruction template:

```python
def chat(model, tokenizer, user_message, max_new_tokens=100):
    # Gemma instruction format
    prompt = f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
    response = generate_text(model, tokenizer, prompt, max_new_tokens)
    return response

# Example conversation
questions = [
    "What is JAX and why would I use it?",
    "Explain tensor parallelism in simple terms.",
    "Write a haiku about machine learning.",
]

for q in questions:
    print(f"User: {q}")
    print(f"Gemma: {chat(model, tokenizer, q)}")
    print()
```

---

## Swapping to a Larger Model

Everything above uses `google/gemma-3-1b-it` (1B parameters). To use a larger model, change the model name:

```python
# 7B model — needs more memory (Colab Pro or multi-device)
model_name = "google/gemma-3-7b-it"
```

The rest of the code remains identical. Larger models produce higher quality outputs but require more memory and compute. The 7B model benefits significantly from tensor parallelism on multi-device setups.

Other models that work well with torchax include any standard HuggingFace `AutoModelForCausalLM` architecture — GPT-2, Llama, Mistral, Phi, and more.

---

## Troubleshooting

**`TypeError: ... is not a valid JAX type`**
You need to register the type as a pytree. See the registration examples above for `CausalLMOutputWithPast`, `DynamicCache`, and `StaticCache`.

**`ConcretizationTypeError: Abstract tracer value encountered`**
A value that changes between calls (like a boolean flag) needs to be either: (1) made static via `static_argnums` in `jax.jit`, or (2) baked into a closure as a constant.

**`UserWarning: A large amount of constants were captured`**
Model weights are being inlined as constants in the compiled graph. Pass them as explicit function arguments instead of closing over them.

**`RuntimeError: No available devices`**
Ensure JAX can see your accelerator: `print(jax.devices())`. In Colab, check that your runtime type is set to TPU or GPU.

---

## Conclusion

In this tutorial, we went from zero to a working chatbot running a HuggingFace model on JAX:

1. **Forward pass** — moved a PyTorch model to JAX with `model.to("jax")`
2. **JIT compilation** — compiled for 10-100x speedup with `jax.jit`
3. **Text classification** — used prompt engineering for sentiment analysis
4. **Text generation** — implemented autoregressive decoding with StaticCache
5. **Distributed inference** — sharded weights across devices with tensor parallelism
6. **Chatbot** — wrapped generation in an instruction-following chat function

The key insight: torchax lets you use the entire HuggingFace ecosystem — models, tokenizers, configs — while running on JAX's high-performance backend. No model rewrites needed.

### What's Next: Fine-Tuning

Ready to go beyond inference? In [Part 2: Fine-Tune Any HuggingFace Model on TPUs with TorchAX](https://dev.to/ahmed_elnaggar/fine-tune-any-huggingface-model-on-tpus-with-torchax), we instruction-tune Gemma 4 E2B using LoRA and torchax's functional training API — all on a free Colab TPU.

### Resources

- [torchax GitHub](https://github.com/google/torchax) — library source and documentation
- [torchax Docs](https://google.github.io/torchax/) — official getting started guide
- [Original tutorial series by Han Qi](https://huggingface.co/blog/qihqi/huggingface-jax-01) — the 3-part blog series this tutorial builds on
- [JAX Documentation](https://docs.jax.dev/) — JIT compilation, pytrees, distributed arrays
- [HuggingFace LLM Inference Optimization](https://huggingface.co/docs/transformers/llm_optims) — StaticCache and torch.compile docs
- [Companion GitHub repo](https://github.com/agemagician/torchax-huggingface) — all code, notebooks, and diagrams

### Credits

This tutorial would not be possible without the work of:
- **Han Qi ([@qihqi](https://github.com/qihqi))** — author of torchax and the [original HuggingFace + JAX tutorial series](https://github.com/qihqi/learning_machine/tree/main/jax-huggingface)
- The **torchax team at Google** — for building and maintaining the library
- The **HuggingFace team** — for the transformers ecosystem
- The **JAX team at Google** — for JAX, XLA, and TPU support

---

What model will you try running on TPUs first? Let me know in the comments!

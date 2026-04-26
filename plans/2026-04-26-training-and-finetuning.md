# Plan: TorchAX Training & Fine-Tuning Extension

**Date:** 2026-04-26
**Status:** Draft - Awaiting User Approval
**Branch:** `feat/training-tutorial` (to be created)

---

## 1. Objective

Extend the torchax-huggingface repository from inference-only to include training capabilities. Deliver a self-contained Colab training notebook, a companion blog post, new architecture diagrams, and updated repo documentation.

---

## 2. Requirements Summary

| # | Requirement | Details |
|---|------------|---------|
| R1 | Training notebook | Load `google/gemma-4-E2B-it`, instruction-tune with torchax on free Colab TPU |
| R2 | Dual training modes | Full fine-tuning AND LoRA/PEFT, user-selectable via config cell |
| R3 | Real dataset | HuggingFace dataset for instruction tuning |
| R4 | Evaluation | Before/after comparison (quantitative + qualitative) |
| R5 | Save & reload | Save fine-tuned model, reload for inference |
| R6 | Blog post | dev.to style, beginner-friendly, step-by-step (same series) |
| R7 | Repo updates | README, diagrams, cross-links between inference and training content |

---

## 3. Research Findings

### 3.1 Model: `google/gemma-4-E2B-it`

**Risk:** This model name may refer to a Gemma 4 variant that is newly released or uses a different Hub ID. The "E2B" likely indicates an efficient/expert 2B-parameter variant.

**Mitigation:** The notebook config cell makes the model name a single variable. If the exact Hub ID differs at implementation time, it can be swapped. Fallback candidates: `google/gemma-2-2b-it` or `google/gemma-3-1b-it`.

**Action:** Verify model existence on HuggingFace Hub before implementation begins.

### 3.2 Memory Budget: Free Colab TPU

Free Colab provides a **TPU v2-8** (8 cores, 8 GB HBM per core, 64 GB total).

| Scenario | Weights | Grads | Optimizer | Activations | Total | Fits? |
|----------|---------|-------|-----------|-------------|-------|-------|
| Full fine-tune (2B, AdamW, bf16) | ~4 GB | ~4 GB | ~8 GB | ~2-4 GB | ~18-20 GB | No (single core) |
| Full fine-tune + FSDP (8 cores) | ~0.5 GB/core | ~0.5 GB/core | ~1 GB/core | ~2-4 GB | ~4-6 GB/core | Yes (tight) |
| Full fine-tune + AdaFactor | ~4 GB | ~4 GB | ~4 GB | ~2-4 GB | ~14-16 GB | Possible with grad ckpt |
| LoRA r=16 (2B, bf16) | ~4 GB frozen | ~50 MB | ~100 MB | ~1-3 GB | ~5-7 GB | Yes (comfortable) |

**Decision:**
- **LoRA** is the default and recommended path. Fits comfortably on a single TPU core.
- **Full fine-tuning** is supported but requires: (a) AdaFactor optimizer instead of AdamW, (b) gradient checkpointing via `jax.checkpoint`, (c) `donate_argnums` for in-place updates, (d) micro-batch=1 with gradient accumulation. If it still OOMs, the notebook catches the error and suggests switching to LoRA.

### 3.3 Dataset Selection

**Recommended: `databricks/databricks-dolly-15k`**

| Property | Value |
|----------|-------|
| Size | 15,011 examples |
| Format | `instruction`, `context`, `response`, `category` |
| License | CC BY-SA 3.0 (commercially usable) |
| Quality | Human-written by Databricks employees |
| Categories | brainstorming, classification, QA, generation, summarization, etc. |

**Why this dataset:**
- Human-written (not GPT-generated) -- important for blog credibility
- Permissive license
- Diverse task categories make for interesting before/after comparisons
- Well-known in the community
- For the demo: train on a 2,000-sample subset for speed (~30-45 min on free TPU)

**Fallback:** `timdettmers/openassistant-guanaco` (Apache 2.0, pre-flattened OASST1 data)

### 3.4 TorchAX Training Architecture

TorchAX training uses a **JAX-native functional approach**, fundamentally different from standard PyTorch:

```
Standard PyTorch:           TorchAX:
  loss = model(x)             jittable = JittableModule(model)
  loss.backward()             params, buffers = jittable.params, jittable.buffers
  optimizer.step()            loss, grads = jax.value_and_grad(loss_fn)(params, ...)
                              updates, opt_state = optax.update(grads, opt_state)
                              params = optax.apply_updates(params, updates)
```

**Key APIs:**
- `torchax.train.JittableModule(model)` -- separates trainable params from frozen buffers
- `torchax.train.make_train_step(model_fn, loss_fn, optimizer)` -- composes the full train step
- `torchax.interop.jax_jit(fn)` -- JIT-compiles a torch function via JAX
- `optax` -- functional optimizers (adam, adafactor, warmup schedules)
- `donate_argnums` -- tells XLA to reuse input memory for outputs (avoids 2x peak memory)

**LoRA integration pattern** (from torchax `peft_lora_training.py`):
1. Load model with `tx.disable_temporarily()` context
2. Apply `peft.get_peft_model(model, LoraConfig(...))`
3. Separate trainable (LoRA) params from frozen (base) params
4. Frozen params treated as buffers (not differentiated)
5. Only LoRA params go through `jax.value_and_grad`

---

## 4. Deliverables

### 4.1 New Files

| File | Description |
|------|-------------|
| `notebooks/torchax_training_tutorial.ipynb` | Main training notebook (Jupyter, self-contained for Colab) |
| `notebooks/torchax_training_quickstart.ipynb` | Condensed training quickstart (~10 cells) |
| `blog/torchax-training-tutorial.md` | dev.to blog post for training/fine-tuning |
| `assets/diagrams/training-loop-flow.md` | Mermaid: functional training loop in torchax |
| `assets/diagrams/lora-architecture.md` | Mermaid: LoRA weight decomposition + JittableModule separation |
| `assets/diagrams/training-vs-inference.md` | Mermaid: side-by-side inference vs training data flows |

### 4.2 Modified Files

| File | Changes |
|------|---------|
| `README.md` | Add training section, Colab badge, update repo structure, add Gemma 4 E2B to models table, update pip install |
| `assets/README.md` | Add 3 new diagram entries to inventory |
| `blog/torchax-huggingface-tutorial.md` | Add "What's Next: Fine-Tuning" section before conclusion |

---

## 5. Implementation Phases

### Phase 0: Validation Spike (Pre-Implementation)

**Goal:** Verify critical assumptions before committing to the full notebook.

| Step | Task | Risk it Validates |
|------|------|-------------------|
| 0.1 | Verify `google/gemma-4-E2B-it` exists on HuggingFace Hub | Model availability |
| 0.2 | Test `peft.get_peft_model()` + `model.to("jax")` compatibility | PEFT + torchax interop |
| 0.3 | Test `torchax.train.JittableModule` API existence and signature | Training API surface |
| 0.4 | Test a single forward + backward pass with LoRA on Colab TPU | End-to-end feasibility |
| 0.5 | Verify Gemma 4 chat template tokens (may differ from Gemma 3) | Data formatting |

**If any spike fails:**
- 0.1 fails: Use `google/gemma-2-2b-it` or `google/gemma-3-1b-it`
- 0.2 fails: Implement manual LoRA (low-rank A/B matrices as separate JAX arrays)
- 0.3 fails: Use `torchax.extract_jax()` + manual `jax.value_and_grad` + optax
- 0.4 fails: Debug op-by-op; if fundamental incompatibility, use `torch_xla` training path
- 0.5 fails: Use `tokenizer.apply_chat_template()` instead of hardcoding tokens

---

### Phase 1: Training Notebook

**File:** `notebooks/torchax_training_tutorial.ipynb`

#### Section A: Setup (Cells 1-5)

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Title: "Fine-Tune Any HuggingFace Model on TPUs with TorchAX". Learning objectives. Colab badge. |
| 2 | Markdown | Background: "Why Fine-Tune on TPUs?" -- training vs inference metaphor, why torchax uses JAX-native approach |
| 3 | Code | **Configuration cell** -- all user-adjustable knobs in one place |

Configuration cell contents:
```python
# === USER CONFIGURATION ===
MODEL_NAME = "google/gemma-4-E2B-it"
TRAINING_MODE = "lora"          # "full" or "lora"

# LoRA settings (ignored if TRAINING_MODE == "full")
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.05

# Dataset
DATASET_NAME = "databricks/databricks-dolly-15k"
MAX_SEQ_LEN = 512
MAX_TRAIN_SAMPLES = 2000        # Subset for Colab time constraints
MAX_EVAL_SAMPLES = 200

# Training
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4            # Effective batch = 8
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4            # Auto-adjusted: 2e-4 for LoRA, 2e-5 for full
WARMUP_STEPS = 50
EVAL_EVERY_STEPS = 100

# Full fine-tuning memory optimizations (ignored if TRAINING_MODE == "lora")
USE_GRADIENT_CHECKPOINTING = True
USE_ADAFACTOR = True            # AdaFactor uses ~50% less memory than AdamW
```

| Cell | Type | Content |
|------|------|---------|
| 4 | Code | Detect accelerator (reuse existing pattern) + install deps: `torchax transformers flax peft datasets optax` |
| 5 | Code | Verify installation, print versions |

#### Section B: Key Concepts (Cells 6-7)

| Cell | Type | Content |
|------|------|---------|
| 6 | Markdown | **"Training with TorchAX: Three New Concepts"** with beginner-friendly metaphors: |

1. **JittableModule** -- "If inference is like reading a book, training is like studying with a highlighter. JittableModule separates the pages (buffers) from the highlighted notes (params) so JAX knows what to learn from."
2. **optax** -- "PyTorch optimizers carry hidden state (momentum, etc.) inside themselves. JAX optimizers are pure functions: you give them gradients and state, they return updates and new state. No hidden pockets."
3. **donate_argnums** -- "Imagine updating a document: instead of copying the whole document, editing the copy, then throwing away the original, `donate_argnums` says 'just edit the original in place.' This halves peak memory."

| Cell | Type | Content |
|------|------|---------|
| 7 | Markdown | **"Full Fine-Tuning vs LoRA"** -- visual explanation with memory comparison table |

#### Section C: Data Preparation (Cells 8-11)

| Cell | Type | Content |
|------|------|---------|
| 8 | Code | Load dataset from HuggingFace Hub, inspect 3 examples |
| 9 | Code | Format examples into Gemma chat template using `tokenizer.apply_chat_template()`. Split train/eval. |
| 10 | Code | Tokenize with padding/truncation to MAX_SEQ_LEN. Create labels. |
| 11 | Code | Create PyTorch DataLoaders with collate function |

Data formatting pattern:
```python
def format_instruction(example):
    messages = [
        {"role": "user", "content": example["instruction"] +
         (f"\n\nContext: {example['context']}" if example.get("context") else "")},
        {"role": "assistant", "content": example["response"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

#### Section D: Model Loading (Cells 12-14)

| Cell | Type | Content |
|------|------|---------|
| 12 | Code | Load base model on CPU (with `tx.disable_temporarily()` or load-before-enable pattern) |
| 13 | Code | Conditionally apply LoRA: `if TRAINING_MODE == "lora": model = get_peft_model(model, lora_config)` + print trainable params |
| 14 | Code | Enable torchax, move model to JAX: `tx.enable_globally(); model.to("jax")` |

**Critical risk:** PEFT + torchax compatibility. If `get_peft_model()` output doesn't work with `model.to("jax")`, fallback to manual LoRA implementation.

#### Section E: Baseline Evaluation (Cells 15-17)

| Cell | Type | Content |
|------|------|---------|
| 15 | Code | Define `evaluate()` function: compute avg cross-entropy loss and perplexity on eval set |
| 16 | Code | Run baseline evaluation, store results. Generate responses to 5 sample prompts. |
| 17 | Code | Display baseline metrics + sample responses in formatted table |

Evaluation metrics:
- Cross-entropy loss on eval set (quantitative)
- Perplexity (derived from loss -- more interpretable for blog readers)
- 5 qualitative before/after examples (most compelling)

#### Section F: Training Setup (Cells 18-20)

| Cell | Type | Content |
|------|------|---------|
| 18 | Code | Create JittableModule, separate params/buffers |

```python
from torchax.train import JittableModule
jittable = JittableModule(model)

if TRAINING_MODE == "lora":
    # Separate trainable LoRA params from frozen base params
    params = {n: p for n, p in jittable.params.items() if "lora" in n.lower()}
    frozen = {n: p for n, p in jittable.params.items() if "lora" not in n.lower()}
    buffers = {**jittable.buffers, **frozen}
else:
    params = jittable.params
    buffers = jittable.buffers
```

| Cell | Type | Content |
|------|------|---------|
| 19 | Code | Create optax optimizer with warmup-cosine schedule |

```python
import optax

lr = LEARNING_RATE if TRAINING_MODE == "lora" else LEARNING_RATE / 10
total_steps = NUM_EPOCHS * len(train_dataloader) // GRAD_ACCUM_STEPS

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=lr,
    warmup_steps=WARMUP_STEPS, decay_steps=total_steps
)

if TRAINING_MODE == "full" and USE_ADAFACTOR:
    optimizer = optax.adafactor(learning_rate=schedule)
else:
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)

opt_state = tx.interop.call_jax(optimizer.init, params)
```

| Cell | Type | Content |
|------|------|---------|
| 20 | Code | Define pure loss function + create JIT-compiled train step |

```python
def model_fn(weights, buffers, batch):
    return torch.func.functional_call(model, {**weights, **buffers},
                                       args=(), kwargs=batch)

def loss_fn(model_output, labels):
    return model_output.loss  # HF models compute CE loss internally

train_step = tx.train.make_train_step(model_fn, loss_fn, optimizer,
    remat_policy=jax.checkpoint_policies.nothing_saveable
    if (TRAINING_MODE == "full" and USE_GRADIENT_CHECKPOINTING) else None)
train_step = tx.interop.jax_jit(train_step,
    kwargs_for_jax_jit={"donate_argnums": (0, 2)})
```

#### Section G: Training Loop (Cells 21-23)

| Cell | Type | Content |
|------|------|---------|
| 21 | Code | Training loop with gradient accumulation, tqdm progress, periodic eval |
| 22 | Code | Plot training loss curve + eval perplexity (matplotlib, 2 subplots) |
| 23 | Markdown | Interpret training curves: what good training looks like, what to do if loss plateaus |

#### Section H: Post-Training Evaluation (Cells 24-26)

| Cell | Type | Content |
|------|------|---------|
| 24 | Code | Final evaluation on full eval set |
| 25 | Code | Generate responses to same 5 prompts from baseline. Side-by-side comparison table. |
| 26 | Code | Print summary metrics: baseline vs trained (loss, perplexity, % improvement) |

#### Section I: Save & Reload (Cells 27-30)

| Cell | Type | Content |
|------|------|---------|
| 27 | Code | Convert JAX params back to CPU tensors, save model (LoRA: adapters only; Full: entire model) |
| 28 | Code | Reload from checkpoint (fresh load, no prior state) |
| 29 | Code | Generate text with reloaded model to verify |
| 30 | Code | Mini chat session (reuse pattern from inference notebook) |

#### Section J: Wrap-Up (Cells 31-33)

| Cell | Type | Content |
|------|------|---------|
| 31 | Markdown | "Swapping Models & Datasets" -- how to use different models/datasets |
| 32 | Markdown | Troubleshooting: OOM, slow training, PEFT compatibility, chat template issues |
| 33 | Markdown | Conclusion, link to inference tutorial, link to blog post, credits |

---

### Phase 2: Mermaid Diagrams

**Can be built in parallel with Phase 1.** Use existing Google color palette: blue (#4285F4), yellow (#FBBC04), green (#34A853), red (#EA4335).

#### 2.1 Training Loop Flow (`assets/diagrams/training-loop-flow.md`)

Flowchart: Model --> JittableModule --> [params] + [buffers] --> loss_fn --> jax.value_and_grad --> optax.update --> apply_updates --> loop. Highlight the JAX boundary.

#### 2.2 LoRA Architecture (`assets/diagrams/lora-architecture.md`)

Two-part diagram: (1) Weight decomposition: W_frozen + (alpha/r) * B @ A. (2) JittableModule separation: base W --> buffers, LoRA A/B --> params.

#### 2.3 Training vs Inference (`assets/diagrams/training-vs-inference.md`)

Side-by-side: Inference (torchax.compile --> jax.jit forward) vs Training (JittableModule --> jax.value_and_grad --> optax). Shared base: torchax.Tensor on TPU.

---

### Phase 3: Blog Post

**File:** `blog/torchax-training-tutorial.md`
**Depends on:** Phase 1 (blog mirrors notebook)

Structure (~4,200 words, matching existing blog length):

```markdown
---
title: "Fine-Tune Any HuggingFace Model on TPUs with TorchAX"
published: false
description: "Instruction-tune HuggingFace models on Google TPUs using torchax -- full fine-tuning and LoRA, with before/after evaluation."
tags: machinelearning, pytorch, python, tutorial
series: "TorchAX + HuggingFace"
---
```

1. **Hook** (~200 words): "What if you could fine-tune any HuggingFace model on TPUs?" Show 10-line end result. Colab badge.
2. **Why This Matters** (~300 words): The training gap -- inference solved in Part 1, training still requires torch-xla or Flax rewrite.
3. **Key Concepts** (~500 words): JittableModule, optax, donate_argnums with metaphors.
4. **Full Fine-Tuning vs LoRA** (~400 words): Visual explanation, memory table, when to use which.
5. **Step-by-Step Tutorial** (~2000 words): Mirrors notebook sections C-I with code and explanations.
6. **Results** (~300 words): Loss curves, perplexity improvement, qualitative examples.
7. **Troubleshooting** (~300 words): Common errors and fixes.
8. **Conclusion** (~200 words): Summary, link to Part 1, credits.

---

### Phase 4: Repo Updates

**Depends on:** Phases 1-3

#### 4.1 README.md

1. Add Colab badge for training notebook
2. Add training items to "What You'll Learn"
3. Add "Training Quick Start" section
4. Update Repository Structure tree
5. Add `google/gemma-4-E2B-it` to Models table
6. Update pip install to include `peft datasets optax`
7. Add training blog post link

#### 4.2 assets/README.md

Add 3 new diagram entries to inventory table.

#### 4.3 Inference Blog Post Cross-Link

Add "What's Next: Fine-Tuning" section before conclusion in existing blog post.

---

## 6. Dependency Graph

```
Phase 0: Validation Spike
  |
  v
Phase 1: Training Notebook -----> Phase 3: Blog Post
  |                                   |
  |  (parallel)                       v
  +---> Phase 2: Diagrams -------> Phase 4: Repo Updates
```

Phase 0 must complete first (validates feasibility).
Phases 1 and 2 can proceed in parallel.
Phase 3 depends on Phase 1 (blog mirrors notebook).
Phase 4 depends on all previous phases (references all new files).

---

## 7. Risks & Mitigations

### Critical

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `google/gemma-4-E2B-it` not on Hub | Medium | High | Config cell makes model swappable. Fallback: `google/gemma-2-2b-it` |
| PEFT + torchax incompatible | Medium | High | Manual LoRA (A/B matrices as JAX arrays). More educational anyway. |
| `torchax.train` API missing/different | Low-Medium | High | Fallback: `extract_jax()` + manual `jax.value_and_grad` + optax |
| Full fine-tuning OOMs on free TPU | High | Medium | Default LoRA. Full uses AdaFactor + grad ckpt + donate_argnums. Graceful error. |

### Medium

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gemma 4 chat template differs from Gemma 3 | Medium | Medium | Use `tokenizer.apply_chat_template()` not hardcoded tokens |
| CE loss through torchax hits unsupported ops | Low | Medium | Compute loss manually via JAX if HF built-in fails |
| Training too slow (session timeout) | Medium | Medium | Default 2000 samples, 1 epoch. Timing estimates in cells. |

---

## 8. Notebook Cell Summary

| # | Section | Type | Content | Risk |
|---|---------|------|---------|------|
| 1 | Setup | MD | Title, objectives, Colab badge | Low |
| 2 | Setup | MD | Background: "Why Fine-Tune on TPUs?" | Low |
| 3 | Setup | Code | Configuration cell (all user knobs) | Low |
| 4 | Setup | Code | Detect accelerator + install deps | Low |
| 5 | Setup | Code | Verify installation | Low |
| 6 | Concepts | MD | JittableModule, optax, donate_argnums | Low |
| 7 | Concepts | MD | Full fine-tuning vs LoRA | Low |
| 8 | Data | Code | Load dataset, inspect examples | Low |
| 9 | Data | Code | Format chat template, split train/eval | Medium |
| 10 | Data | Code | Tokenize, create labels | Low |
| 11 | Data | Code | Create DataLoaders | Low |
| 12 | Model | Code | Load base model on CPU | Medium |
| 13 | Model | Code | Conditionally apply LoRA | **High** |
| 14 | Model | Code | Enable torchax, move to JAX | Medium |
| 15 | Eval | Code | Define evaluate() function | Medium |
| 16 | Eval | Code | Run baseline evaluation | Low |
| 17 | Eval | Code | Display baseline metrics | Low |
| 18 | Train | Code | JittableModule, separate params/buffers | **High** |
| 19 | Train | Code | optax optimizer with schedule | Low |
| 20 | Train | Code | loss_fn + JIT-compiled train_step | **High** |
| 21 | Train | Code | Training loop with grad accumulation | Medium |
| 22 | Train | Code | Plot loss curves | Low |
| 23 | Train | MD | Interpret training curves | Low |
| 24 | Results | Code | Final evaluation | Low |
| 25 | Results | Code | Side-by-side comparison | Low |
| 26 | Results | Code | Summary metrics table | Low |
| 27 | Save | Code | Convert JAX params, save model | Medium |
| 28 | Save | Code | Reload from checkpoint | Low |
| 29 | Save | Code | Generate with reloaded model | Low |
| 30 | Save | Code | Mini chat session | Low |
| 31 | Wrap | MD | Swapping models & datasets | Low |
| 32 | Wrap | MD | Troubleshooting guide | Low |
| 33 | Wrap | MD | Conclusion, links, credits | Low |

**Total:** 33 cells (18 code, 15 markdown). High-risk cells: 13, 18, 20.

---

## 9. Success Criteria

- [ ] Training notebook runs end-to-end on free Colab TPU (LoRA mode)
- [ ] Training loss decreases over the run
- [ ] Eval perplexity improves by at least 10% vs baseline
- [ ] At least 3 qualitative before/after examples show visible improvement
- [ ] Fine-tuned model saves to disk and reloads for inference
- [ ] Full fine-tuning either works or fails gracefully with clear message
- [ ] Blog post renders correctly on dev.to
- [ ] All 3 Mermaid diagrams render at mermaid.live
- [ ] README reflects updated repo structure
- [ ] Cross-links between inference and training content work
- [ ] Colab badge opens correct notebook
- [ ] Total training time < 60 min on free Colab TPU (LoRA, 2000 samples)

---

## 10. Estimated Effort

| Phase | Steps | Effort | Risk |
|-------|-------|--------|------|
| Phase 0: Validation Spike | 5 checks | 0.5 day | High (determines feasibility) |
| Phase 1: Training Notebook | 33 cells | 3-4 days | High (torchax API unknowns) |
| Phase 2: Mermaid Diagrams | 3 files | 0.5 day | Low |
| Phase 3: Blog Post | ~4200 words | 1-1.5 days | Low |
| Phase 4: Repo Updates | 3 file edits | 0.5 day | Low |
| **Total** | | **5.5-7 days** | **Medium-High** |

---

## 11. Resolved Questions

1. **Model name:** `google/gemma-4-E2B-it` confirmed as correct Hub ID by user.
2. **Full fine-tuning priority:** LoRA is the primary/default path. Full fine-tuning is best-effort.
3. **Training quickstart notebook:** Yes -- create BOTH a comprehensive tutorial AND a condensed quickstart (like existing `torchax_quickstart.ipynb`).
4. **Colab testing:** Yes -- test on Colab during implementation.

---

## 12. Additional Deliverable: Training Quickstart Notebook

**File:** `notebooks/torchax_training_quickstart.ipynb`

Condensed version of the training tutorial (~8-10 cells), mirroring how `torchax_quickstart.ipynb` relates to `torchax_huggingface_tutorial.ipynb`.

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Title, Colab badge, one-paragraph overview |
| 2 | Code | Install deps + detect accelerator |
| 3 | Code | Config cell (model, mode, dataset, hyperparams) |
| 4 | Code | Load dataset, format, tokenize (all in one cell) |
| 5 | Code | Load model + apply LoRA + move to JAX |
| 6 | Code | Setup training (JittableModule + optax + train_step) |
| 7 | Code | Training loop (compact, with tqdm) |
| 8 | Code | Evaluate + print before/after metrics |
| 9 | Code | Save model + reload + generate |
| 10 | Markdown | Links to full tutorial and blog post |

This adds to Phase 1 and the quickstart should be built AFTER the full tutorial is validated.

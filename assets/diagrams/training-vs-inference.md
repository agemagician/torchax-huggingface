# Training vs Inference Comparison Diagram

Render at https://mermaid.live or with `mmdc` CLI.

```mermaid
graph TB
    START["PyTorch HuggingFace Model<br/>model.to('jax')"]

    subgraph "Inference Path"
        I1["torchax.compile(model)"]
        I2["jax.jit(forward)"]
        I3["input_ids → output logits"]
    end

    subgraph "Training Path"
        T1["JittableModule(model)"]
        T2["Separate params / buffers"]
        T3["jax.value_and_grad(loss_fn)"]
        T4["optax optimizer"]
        T5["Updated params"]
    end

    subgraph "Shared Foundation"
        S1["torchax.Tensor wraps jax.Array"]
        S2["XLA execution on TPU / GPU"]
    end

    START --> I1
    START --> T1
    I1 --> I2
    I2 --> I3
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> T5
    T5 -->|"next step"| T3
    I3 --> S1
    T5 --> S1
    S1 --> S2

    style START fill:#fff,stroke:#333,color:#333
    style I1 fill:#4285F4,stroke:#333,color:#fff
    style I2 fill:#4285F4,stroke:#333,color:#fff
    style I3 fill:#4285F4,stroke:#333,color:#fff
    style T1 fill:#34A853,stroke:#333,color:#fff
    style T2 fill:#34A853,stroke:#333,color:#fff
    style T3 fill:#34A853,stroke:#333,color:#fff
    style T4 fill:#34A853,stroke:#333,color:#fff
    style T5 fill:#34A853,stroke:#333,color:#fff
    style S1 fill:#FBBC04,stroke:#333,color:#333
    style S2 fill:#FBBC04,stroke:#333,color:#333
```

## Text Description

```
                  ┌──────────────────────────────┐
                  │  PyTorch HuggingFace Model    │
                  │  model.to("jax")              │
                  └──────────┬─────────┬──────────┘
                             │         │
              ┌──────────────┘         └──────────────┐
              ▼                                       ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  Inference Path               │  │  Training Path                │
│                               │  │                               │
│  torchax.compile(model)      │  │  JittableModule(model)        │
│       │                       │  │       │                       │
│       ▼                       │  │       ▼                       │
│  jax.jit(forward)            │  │  Separate params / buffers    │
│       │                       │  │       │                       │
│       ▼                       │  │       ▼                       │
│  input → output logits       │  │  jax.value_and_grad(loss_fn)  │
│                               │  │       │                       │
│                               │  │       ▼                       │
│                               │  │  optax optimizer              │
│                               │  │       │                       │
│                               │  │       ▼                       │
│                               │  │  updated params ─→ (loop)    │
└──────────────┬───────────────┘  └──────────────┬───────────────┘
               │                                  │
               └──────────────┬───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Shared Foundation                                               │
│                                                                   │
│  torchax.Tensor wraps jax.Array                                  │
│  Both paths use the same dispatch mechanism:                     │
│    torch op → __torch_dispatch__ → JAX equivalent                │
│  XLA execution on TPU / GPU / CPU                                │
└─────────────────────────────────────────────────────────────────┘
```

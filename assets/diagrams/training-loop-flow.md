# Training Loop Flow Diagram

Render at https://mermaid.live or with `mmdc` CLI.

```mermaid
graph TB
    subgraph "PyTorch Side"
        A["nn.Module<br/>(HuggingFace model)"]
        B["JittableModule<br/>(functional wrapper)"]
    end

    subgraph "torchax Bridge"
        C["Separate params<br/>(trainable weights)"]
        D["Separate buffers<br/>(frozen / non-differentiable)"]
    end

    subgraph "JAX Side"
        E["loss_fn(params, buffers, batch)"]
        F["jax.value_and_grad(loss_fn)"]
        G["loss + gradients"]
        H["optax.update(grads, opt_state)"]
        I["optax.apply_updates<br/>(params, updates)"]
        J["new_params"]
    end

    A -->|"JittableModule(model)"| B
    B -->|"jm.params"| C
    B -->|"jm.buffers"| D
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J -->|"next step"| E

    style A fill:#4285F4,stroke:#333,color:#fff
    style B fill:#4285F4,stroke:#333,color:#fff
    style C fill:#FBBC04,stroke:#333,color:#333
    style D fill:#FBBC04,stroke:#333,color:#333
    style E fill:#34A853,stroke:#333,color:#fff
    style F fill:#34A853,stroke:#333,color:#fff
    style G fill:#34A853,stroke:#333,color:#fff
    style H fill:#34A853,stroke:#333,color:#fff
    style I fill:#34A853,stroke:#333,color:#fff
    style J fill:#34A853,stroke:#333,color:#fff
```

## Text Description

```
┌─────────────────────────────────────────────────┐
│  PyTorch Side                                    │
│                                                   │
│  nn.Module (HuggingFace model)                   │
│       │                                           │
│       ▼                                           │
│  JittableModule (functional wrapper)             │
└───────────┬───────────────┬─────────────────────┘
            │               │
            ▼               ▼
┌─────────────────────────────────────────────────┐
│  torchax Bridge                                   │
│                                                   │
│  params (trainable)    buffers (frozen)           │
└───────────┬───────────────┬─────────────────────┘
            │               │
            ▼               ▼
┌─────────────────────────────────────────────────┐
│  JAX Side (functional training loop)             │
│                                                   │
│  loss_fn(params, buffers, batch)                 │
│       │                                           │
│       ▼                                           │
│  jax.value_and_grad(loss_fn)                     │
│       │                                           │
│       ▼                                           │
│  loss + gradients                                │
│       │                                           │
│       ▼                                           │
│  optax.update(grads, opt_state) → updates        │
│       │                                           │
│       ▼                                           │
│  optax.apply_updates(params, updates)            │
│       │                                           │
│       ▼                                           │
│  new_params ──────────────────────→ (loop back)  │
└─────────────────────────────────────────────────┘
```

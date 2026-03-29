# TorchAX Architecture Diagram

Render at https://mermaid.live or with `mmdc` CLI.

```mermaid
graph TB
    subgraph "What You Write"
        A["PyTorch Model<br/>(torch.nn.Module)"]
        B["model(input_ids)"]
    end

    subgraph "What TorchAX Does (The Trojan Horse)"
        C["torchax.Tensor<br/>isinstance(t, torch.Tensor) == True"]
        D["__torch_dispatch__<br/>intercepts every PyTorch op"]
        E["Maps torch op → JAX equivalent"]
    end

    subgraph "What Actually Runs"
        F["jax.Array<br/>(stored inside torchax.Tensor)"]
        G["JAX Execution<br/>on TPU / GPU / CPU"]
    end

    A -->|"model.to('jax')"| C
    B --> D
    D --> E
    E --> F
    F --> G

    style A fill:#4285F4,stroke:#333,color:#fff
    style B fill:#4285F4,stroke:#333,color:#fff
    style C fill:#FBBC04,stroke:#333,color:#333
    style D fill:#FBBC04,stroke:#333,color:#333
    style E fill:#FBBC04,stroke:#333,color:#333
    style F fill:#34A853,stroke:#333,color:#fff
    style G fill:#34A853,stroke:#333,color:#fff
```

## Text Description

```
┌─────────────────────────────────────────────────┐
│  Your Code (unchanged PyTorch)                   │
│                                                   │
│  model = AutoModelForCausalLM.from_pretrained()  │
│  model.to("jax")   ← only change                │
│  output = model(input_ids)                       │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  torchax Layer (The Trojan Horse)                │
│                                                   │
│  torchax.Tensor wraps jax.Array                  │
│  PyTorch sees: torch.Tensor ✓                    │
│  Actually contains: jax.Array                    │
│                                                   │
│  Every torch op → intercepted → JAX equivalent   │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  JAX Backend                                      │
│                                                   │
│  Eager mode: ops execute one at a time           │
│  JIT mode:   ops compiled + fused by XLA         │
│  Hardware:   TPU / GPU / CPU                     │
└─────────────────────────────────────────────────┘
```

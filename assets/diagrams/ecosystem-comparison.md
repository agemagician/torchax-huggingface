# Ecosystem Comparison Decision Tree

Render at https://mermaid.live or with `mmdc` CLI.

```mermaid
graph TD
    START["I have a PyTorch/HuggingFace model.<br/>I want to run it on TPUs or use JAX."]

    Q1{"Willing to rewrite<br/>the model in JAX?"}
    Q2{"Need a full PyTorch<br/>training loop on TPU?"}
    Q3{"Need JAX interop?<br/>(jax.jit, jax.grad, gSPMD)"}
    Q4{"Deploying to<br/>non-Python runtime?"}

    A1["Flax / Equinox<br/>Native JAX performance<br/>Full ecosystem access"]
    A2["torch-xla (PyTorch/XLA)<br/>Minimal code changes<br/>PyTorch training on TPU"]
    A3["torchax<br/>Zero model changes<br/>JAX JIT + interop + TPU"]
    A4["ONNX Export<br/>Framework-agnostic<br/>Variable performance"]
    A5["torchax<br/>Best of both worlds"]

    START --> Q1
    Q1 -->|"Yes"| A1
    Q1 -->|"No"| Q2
    Q2 -->|"Yes"| A2
    Q2 -->|"No"| Q3
    Q3 -->|"Yes"| A3
    Q3 -->|"No"| Q4
    Q4 -->|"Yes"| A4
    Q4 -->|"No"| A5

    style START fill:#fff,stroke:#333,color:#333
    style A1 fill:#4285F4,stroke:#333,color:#fff
    style A2 fill:#FBBC04,stroke:#333,color:#333
    style A3 fill:#34A853,stroke:#333,color:#fff
    style A4 fill:#EA4335,stroke:#333,color:#fff
    style A5 fill:#34A853,stroke:#333,color:#fff
```

## Comparison Table

```
┌─────────────────────┬──────────┬─────────────┬──────────────────────────────┐
│ Approach            │ Effort   │ Performance │ Best For                      │
├─────────────────────┼──────────┼─────────────┼──────────────────────────────┤
│ Flax / Equinox      │ High     │ Native JAX  │ New projects in JAX           │
│ torch-xla           │ Low      │ Good (XLA)  │ PyTorch training on TPUs      │
│ torchax             │ Low      │ Great (JIT) │ HF models on JAX, interop     │
│ ONNX Export         │ Medium   │ Variable    │ Cross-framework deployment    │
└─────────────────────┴──────────┴─────────────┴──────────────────────────────┘

Key differentiator:
- torch-xla: Optimizes PyTorch training on XLA devices
- torchax:   Enables JAX-native features (jax.jit, jax.grad, gSPMD) for PyTorch models
```

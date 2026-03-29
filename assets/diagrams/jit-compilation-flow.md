# JIT Compilation Flow Diagram

Render at https://mermaid.live or with `mmdc` CLI.

```mermaid
graph LR
    subgraph "Eager Mode (Without JIT)"
        E1["Python"] --> E2["matmul"]
        E2 --> E3["Python"]
        E3 --> E4["relu"]
        E4 --> E5["Python"]
        E5 --> E6["linear"]
        E6 --> E7["Python"]
    end

    subgraph "JIT Mode (With jax.jit)"
        J1["Python<br/>(first call only)"] --> J2["Trace"]
        J2 --> J3["XLA Compile"]
        J3 --> J4["Fused Kernel:<br/>matmul+relu+linear"]
        J4 --> J5["Result"]
    end

    style E1 fill:#EA4335,stroke:#333,color:#fff
    style E3 fill:#EA4335,stroke:#333,color:#fff
    style E5 fill:#EA4335,stroke:#333,color:#fff
    style E7 fill:#EA4335,stroke:#333,color:#fff
    style E2 fill:#4285F4,stroke:#333,color:#fff
    style E4 fill:#4285F4,stroke:#333,color:#fff
    style E6 fill:#4285F4,stroke:#333,color:#fff
    style J1 fill:#FBBC04,stroke:#333,color:#333
    style J2 fill:#FBBC04,stroke:#333,color:#333
    style J3 fill:#FBBC04,stroke:#333,color:#333
    style J4 fill:#34A853,stroke:#333,color:#fff
    style J5 fill:#34A853,stroke:#333,color:#fff
```

## Timeline Comparison

```
Eager Mode (every call):
┌──────┐ ┌────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌────────┐ ┌──────┐
│Python│→│ matmul │→│Python│→│ relu │→│Python│→│ linear │→│Python│
└──────┘ └────────┘ └──────┘ └──────┘ └──────┘ └────────┘ └──────┘
←─────────────── ~500ms per call ──────────────────→

JIT Mode (first call):
┌──────┐ ┌───────┐ ┌─────────┐ ┌─────────────────────────┐
│Python│→│ Trace │→│ Compile │→│ Execute fused kernel    │
└──────┘ └───────┘ └─────────┘ └─────────────────────────┘
←────────── ~4000ms (one-time) ────────────────────→

JIT Mode (subsequent calls):
┌─────────────────────────┐
│ Execute fused kernel    │
└─────────────────────────┘
←──── ~5ms ────→
```

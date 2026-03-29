# Assets

## Diagrams

The diagrams in `diagrams/` are written in [Mermaid](https://mermaid.js.org/) syntax and can be rendered to SVG/PNG.

### Rendering Options

**Online (recommended):**
1. Go to [mermaid.live](https://mermaid.live/)
2. Paste the Mermaid code block from any `.md` file
3. Export as SVG or PNG

**CLI:**
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i diagrams/torchax-architecture.md -o torchax-architecture.svg
```

**GitHub:** Mermaid diagrams render natively in GitHub markdown.

## Diagram Inventory

| File | Description |
|------|-------------|
| `diagrams/torchax-architecture.md` | How torchax bridges PyTorch and JAX |
| `diagrams/jit-compilation-flow.md` | Eager vs JIT compilation comparison |
| `diagrams/tensor-parallelism.md` | Weight sharding for distributed inference |
| `diagrams/ecosystem-comparison.md` | Decision tree: torchax vs alternatives |

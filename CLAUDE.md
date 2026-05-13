# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Binary image denoising for MNIST digits using a Bayesian Ising Markov Random Field inferred with Infer.NET (EP). The design spec is in `ising_mnist_infernet.md`; the implementation lives in the `.cs` files at the repo root.

**Goal:** Given a corrupted binary MNIST image Y, infer the posterior P(X | Y) and produce a denoised MAP reconstruction.

## Build and Run

```bash
dotnet build
dotnet run                  # digit 8, sample 0 (defaults)
dotnet run -- 3 12          # digit 3, sample index 12
```

First run compiles the Infer.NET factor graph (10–30 s). Subsequent runs reuse the compiled algorithm.

## Model Architecture

Two components in `IsingModel.cs`:

1. **Ising prior** — `VariableArray2D<bool> X` declared with `Variable.Array<bool>(rows, cols)` and a flat `Bernoulli(0.5)` plate prior. 4-connected neighbours are coupled by `ConstrainNeighbours`, which encodes the symmetric potential via `Variable.ConstrainEqualRandom(b, new Bernoulli(pAgree))` inside `Variable.If/IfNot(a)` blocks. `pAgree = sigmoid(2β)`.

2. **Noise model** — `VariableArray2D<bool> Y` declared with `Variable.Observed(...)`. Likelihood factors are attached inside a `Variable.ForEach(rows, cols)` plate using `Variable.ConstrainEqualRandom` conditioned on X.

Inference: `engine.Infer<Bernoulli[,]>(X)` — one call returns all pixel marginals. MAP reconstruction is a 0.5 threshold on the marginals.

## Key Parameters

| Symbol | Role | Typical range |
|--------|------|---------------|
| β | Ising coupling strength | 0.5–2.5; EP may diverge above ~2.5 |
| ε | Per-pixel noise flip probability | 0.05–0.30; must be < 0.5 |

## Known Difficulties

- Loopy inference — no exact solution on the 2D grid; EP is approximate.
- EP convergence degrades near the phase transition (high β).
- Scaling: 28×28 = 784 variables, ~1500 edges; crop to a smaller region for fast iteration.

## Infer.NET Reference Repository

The Infer.NET source is available locally at **`~/git/infer`** (the `dotnet/infer` GitHub repository). Consult it whenever you need to:

- Verify or discover correct API usage (tutorials in `src/Tutorials/`, examples in `src/Examples/`).
- Understand message-passing schedules or generated-code structure (`src/Compiler/`).
- Check whether a bug has been fixed or a new feature added — grep the changelog or git log in that repo first before working around a suspected framework limitation.

### Patterns confirmed from that codebase

| Pattern | Where to find it |
|---------|-----------------|
| `VariableArray2D` declared with `Variable.Array<T>(rows, cols)` | `src/Tutorials/BugsRats.cs`, `src/Tutorials/BayesianPCA.cs` |
| Plate assignment `arr[r, c] = expr.ForEach(r, c)` | `src/Tutorials/BayesianPCA.cs` |
| `Variable.If` / `Variable.IfNot` with `SetTo` or `ConstrainEqualRandom` | `src/Examples/Crowdsourcing/BCC.cs` |
| Inferring a whole array at once: `engine.Infer<Dist[]>(arr)` | `src/Examples/Crowdsourcing/BCC.cs` lines 258–260 |
| `Engine.Compiler.WriteSourceFiles = false` to suppress disk writes | `src/Examples/Crowdsourcing/BCC.cs` line 186 |

### `Range` naming conflict

`Microsoft.ML.Probabilistic.Models.Range` conflicts with `System.Range` (implicit usings). Resolve with a file-level alias:

```csharp
using Range = Microsoft.ML.Probabilistic.Models.Range;
```

### Generated source

Infer.NET compiles the factor graph to C# on the first `Infer()` call. `WriteSourceFiles` is set to `false` in this project so nothing is written to disk. If you temporarily enable it for debugging, the output lands in `GeneratedSource/` which is git-ignored.

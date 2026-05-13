# Infer.NET Ising Model — MNIST Denoising

Bayesian binary image restoration using an Ising Markov Random Field and Infer.NET.

## Problem

Given a corrupted binary MNIST image $Y \in \{0,1\}^{28 \times 28}$, recover the clean latent image $X$ by inferring the posterior $P(X \mid Y)$. Each pixel may have been independently flipped during corruption, and the spatial structure of handwritten digits (connected strokes, smooth regions) provides a strong signal for recovery.

## Why Bayesian Inference?

A discriminative model trained to denoise would need labelled (noisy, clean) pairs and would produce a single point estimate. The Bayesian approach here:

- **Needs no training data** — the prior and noise model are specified analytically.
- **Quantifies uncertainty** — outputs a full posterior, not just a single reconstruction, so per-pixel confidence is available alongside the MAP estimate.
- **Is interpretable** — every parameter ($\beta$, $\varepsilon$) has a direct physical meaning and can be set from domain knowledge or estimated from data.

## Model

The joint distribution factorises as:

$$P(X, Y) = P(X) \prod_{i,j} P(Y_{i,j} \mid X_{i,j})$$

**Ising prior** — neighbouring pixels are coupled with strength $\beta$:

$$P(X) \propto \exp\!\left(\beta \sum_{(i,j)\sim(k,l)} X_{i,j} \cdot X_{k,l}\right)$$

Higher $\beta$ → smoother reconstructions. The 4-neighbourhood (up/down/left/right) is used.

**Noise model** — binary symmetric channel with flip probability $\varepsilon$:

$$P(Y_{i,j} \mid X_{i,j}) = \begin{cases}
1 - \varepsilon & \text{if } Y_{i,j} = X_{i,j} \\
\varepsilon & \text{otherwise}
\end{cases}$$

**Inference** is performed with Infer.NET using Expectation Propagation (EP), producing marginals $P(X_{i,j} = 1 \mid Y)$ and a MAP reconstruction.

## Implementation

- `IsingModel.cs` — the Infer.NET factor graph. X (latent) and Y (observed) are both `VariableArray2D<bool>`. The Ising coupling is encoded as soft equality constraints via `Variable.ConstrainEqualRandom`. Inference runs a single `engine.Infer<Bernoulli[,]>(X)` call to retrieve all pixel marginals at once.
- `MnistLoader.cs` — IDX binary format reader and per-label index.
- `ImageUtils.cs` — noise injection, cropping, thresholding, and console rendering.
- `Program.cs` — CLI entry point; accepts `[digit] [index]` arguments.

## Build and Run

Prerequisites: [.NET SDK](https://dotnet.microsoft.com/download) (tested on .NET 10).

```bash
dotnet build
dotnet run                  # digit 8, sample 0 (defaults)
dotnet run -- 3 12          # digit 3, sample index 12
```

The first run compiles the factor graph (10–30 s); subsequent runs reuse the compiled algorithm and are much faster.

## Data

The project uses binary MNIST images. Download instructions and the dataset are available at:

https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Place the four IDX files under a `data/` directory before running:

```
data/
  train-images.idx3-ubyte
  train-labels.idx1-ubyte
  t10k-images.idx3-ubyte
  t10k-labels.idx1-ubyte
```

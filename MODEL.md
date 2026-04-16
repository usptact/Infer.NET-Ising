# MODEL.md — Ising Denoising: C# Implementation

This document describes the implemented Ising MRF image denoising model. The probabilistic specification is in `ising_mnist_infernet.md`.

---

## 1. Math → Infer.NET Mapping

### 1.1 Ising Prior — Symmetric Pairwise Coupling

The spec's energy `β Σ X_i X_j` only rewards (1,1) pairs, which is asymmetric. The implementation uses the symmetric formulation that rewards *agreement* between neighbours:

```
ψ(X_i, X_j) = exp(β · (2X_i − 1)(2X_j − 1))
```

This gives:
- Same value (0,0) or (1,1): weight exp(+β)
- Different values (0,1) or (1,0): weight exp(−β)

Normalised:

```
P(X_i = X_j) = exp(β) / (exp(β) + exp(−β)) = sigmoid(2β)
```

**Infer.NET encoding.** `Variable.AreEqual` is not available in 0.4.x. The pairwise factor is encoded as a conditional likelihood using gate blocks:

```csharp
double pAgree = 1.0 / (1.0 + Math.Exp(-2.0 * beta));  // sigmoid(2β)

// For each edge (a, b):
using (Variable.If(a))
    Variable.ConstrainEqualRandom(b, new Bernoulli(pAgree));
using (Variable.IfNot(a))
    Variable.ConstrainEqualRandom(b, new Bernoulli(1.0 - pAgree));
```

The two gate blocks together produce the joint factor:

| a | b | factor |
|---|---|--------|
| 1 | 1 | pAgree |
| 1 | 0 | 1 − pAgree |
| 0 | 1 | 1 − pAgree |
| 0 | 0 | pAgree |

### 1.2 Observation Model — Binary Symmetric Noise

```
P(Y_ij | X_ij) = (1 − ε)  if Y_ij = X_ij
                    ε       if Y_ij ≠ X_ij
```

**Infer.NET encoding:**

```csharp
Variable<bool> y = Variable.New<bool>();
using (Variable.If(x))    y.SetTo(Variable.Bernoulli(1.0 - epsilon));
using (Variable.IfNot(x)) y.SetTo(Variable.Bernoulli(epsilon));
y.ObservedValue = observedPixel;   // set in Infer() before running EP
```

### 1.3 Prior on Each Pixel

The spatial structure is fully encoded by the pairwise coupling factors, so each pixel gets a flat marginal prior:

```csharp
_x[i, j] = Variable.Bernoulli(0.5).Named($"x{i}_{j}");
```

### 1.4 Posterior Extraction

```csharp
Bernoulli post  = engine.Infer<Bernoulli>(_x[i, j]);
double   probOn = post.GetProbTrue();   // P(X_ij = 1 | Y)
bool     map    = probOn >= 0.5;        // MAP estimate
```

---

## 2. File Layout

```
infernet-ising.csproj
Program.cs      ← CLI entry point (digit + index args, MNIST loading, demo flow)
IsingModel.cs   ← factor graph construction and EP inference
ImageUtils.cs   ← noise, cropping, MAP thresholding, console rendering
MnistLoader.cs  ← IDX file loading and binarization (see DATA.md)
```

---

## 3. `IsingModel.cs`

### Class signature

```csharp
public sealed class IsingModel
{
    public int Height { get; }
    public int Width  { get; }

    // Build the factor graph once; call Infer() for each image.
    public IsingModel(int height, int width, double beta, double epsilon,
                      int iterations = 30) { … }

    // Attach observations, run EP, return P(X_ij=1 | Y) for every pixel.
    public double[,] Infer(bool[,] noisyImage) { … }
}
```

### Fields

```csharp
private readonly Variable<bool>[,] _x;   // latent clean-image pixels  (H × W)
private readonly Variable<bool>[,] _y;   // observed noisy-image pixels (H × W)
private readonly InferenceEngine   _engine;
```

`Variable<bool>[,]` is a plain C# rectangular array of `Variable<bool>` references. `VariableArray2D` is not used because `.ForEach` only supports per-element independent factors; pairwise coupling across neighbours requires explicit loops over individual variable references.

### Constructor — factor graph construction

```
1. DeclareLatentVariables()
   for each (i, j):
       _x[i, j] = Variable.Bernoulli(0.5).Named($"x{i}_{j}")

2. AddIsingCoupling(beta)
   pAgree = sigmoid(2β)
   Horizontal edges (i,j) ↔ (i, j+1):
       ConstrainNeighbours(_x[i,j], _x[i,j+1], pAgree)
   Vertical edges (i,j) ↔ (i+1, j):
       ConstrainNeighbours(_x[i,j], _x[i+1,j], pAgree)

3. AddObservationModel(epsilon)
   for each (i, j):
       _y[i,j] = Variable.New<bool>().Named($"y{i}_{j}")
       using (Variable.If(_x[i,j]))    _y[i,j].SetTo(Variable.Bernoulli(1−ε))
       using (Variable.IfNot(_x[i,j])) _y[i,j].SetTo(Variable.Bernoulli(ε))

4. Create engine
   new InferenceEngine(new ExpectationPropagation())
       { NumberOfIterations = iterations, ShowProgress = false }
```

### `Infer` method

```
1. Set observations:
   for each (i, j): _y[i,j].ObservedValue = noisyImage[i,j]

2. Query marginals:
   for each (i, j):
       posteriors[i,j] = engine.Infer<Bernoulli>(_x[i,j]).GetProbTrue()

3. Return posteriors[H, W]
```

**Performance note.** With individually named `Variable<bool>` objects, Infer.NET re-runs EP for each `Infer` call rather than caching results across calls. For a 10×10 grid this means 100 EP runs (~80 s total). The fix is to move to a `VariableArray`-based model that supports `engine.Infer<Bernoulli[,]>(xArray)` in a single pass; this is future work.

---

## 4. `ImageUtils.cs`

Static helper methods:

| Method | Signature | Description |
|--------|-----------|-------------|
| `CropCenter` | `(bool[,], int size) → bool[,]` | Extract centred `size×size` region |
| `AddBinaryNoise` | `(bool[,], double ε, Random?) → bool[,]` | Flip each pixel with prob ε |
| `Threshold` | `(double[,], double t=0.5) → bool[,]` | MAP estimate from posteriors |
| `CountErrors` | `(bool[,], bool[,]) → int` | Pixel-level error count |
| `PrintBinary` | `(bool[,], string?)` | Console render: `█` / space |
| `PrintPosteriors` | `(double[,], string?)` | Shaded render: space/░/▒/▓/█ |
| `PrintStats` | `(double[,], string?)` | Print min / max / mean |

---

## 5. `Program.cs` — Demo Flow

```
Usage: infernet-ising [digit] [index]
  digit   0–9 (default: 8)
  index   sample index within that digit class (default: 0)
```

Runtime flow:

```
1. Parse and validate CLI arguments (format check before loading)
2. Load MNIST training set from data/
3. Validate digit exists and index is in range (with count shown on error)
4. Binarize the selected sample, crop the centre 10×10 region
5. Add binary symmetric noise (ε = 0.15, seed = 42)
6. Display clean and noisy crops
7. Build IsingModel(10, 10, β=0.5, ε=0.15)
8. Run EP inference, time it
9. Print posterior stats, MAP denoised image, shaded posterior grid
```

---

## 6. Parameter Guidance

| Parameter | Working value | Notes |
|-----------|--------------|-------|
| β (coupling) | **0.5** | Safe below EP phase transition (see §7). Do not exceed ~0.44 on a 4-connected grid. |
| ε (noise level) | 0.05 – 0.30 | Must match the rate used when adding noise |
| `NumberOfIterations` | 30 | Increase to 50 for marginal accuracy; decrease for speed |

---

## 7. Known Constraints and Gotchas

### EP phase transition on 4-connected grids

For a 4-connected grid, EP's Bethe-lattice approximation has a ferromagnetic phase transition at:

```
pAgree_critical = (1 + 1/√(degree − 1)) / 2 = (1 + 1/√3) / 2 ≈ 0.789
```

Above this threshold EP collapses to an all-true or all-false fixed point, completely ignoring the observation likelihood. With `pAgree = sigmoid(2β)`, the safe upper bound on β is:

```
β_max = atanh(1/√3) / 2 ≈ 0.44   (for degree-4 grid)
```

β = 0.5 gives pAgree ≈ 0.731, safely below 0.789.

### One EP run per `Infer` call

When using `Variable<bool>[,]` (plain C# arrays of individual variables), Infer.NET does not cache EP results across separate `Infer(variable)` calls. Each call re-runs all message-passing iterations. For a 10×10 grid with 30 iterations, 100 `Infer` calls ≈ 80 s. Mitigation: switch to `VariableArray`-based model allowing a single `Infer<Bernoulli[,]>` call.

### Model compilation on first `Infer` call

The first call generates C# source to `GeneratedSource/`, compiles it with Roslyn, and JIT-compiles the result. This takes roughly 5–10 s regardless of grid size. Subsequent runs of the **same program** (same grid, same β/ε) reuse the cached compiled DLL.

### `Variable.AreEqual` not available

`Variable.AreEqual` does not exist in `Microsoft.ML.Probabilistic` 0.4.x. Use the `Variable.If` / `Variable.IfNot` gate pattern shown in §1.1.

### Grid size limits with individual variables

On a 28×28 grid (784 variables, ~5 000 factors), Infer.NET's Roslyn-based code generation exceeds available memory during compilation (OOM / exit code 137). The current implementation is validated on 10×10 crops. Scaling to full MNIST images requires the `VariableArray` refactor.

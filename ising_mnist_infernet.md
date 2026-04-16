# Ising Model for MNIST Denoising using Infer.NET

## 1. Problem Statement

We observe a corrupted binary MNIST image:

- Observed image: \( Y \in \{0,1\}^{28 \times 28} \)
- True (latent) clean image: \( X \in \{0,1\}^{28 \times 28} \)

Goal:

\[
\text{Infer } P(X \mid Y)
\]

and recover a denoised version of the image using Bayesian inference.

This is a **binary image restoration problem** with spatial structure.

---

## 2. Key Modeling Idea

We assume two principles:

### (A) Spatial coherence (Ising prior)
Neighboring pixels tend to agree.

### (B) Local corruption (noise model)
Each pixel is independently flipped with some probability.

This leads to a **Markov Random Field (MRF)**:

- Latent field: Ising model over pixels
- Observation model: noisy channel per pixel

---

## 3. Latent Model: Ising Prior

Let each pixel be a binary random variable:

\[
X_{i,j} \in \{0,1\}
\]

Define neighborhood \( \mathcal{N}(i,j) \) (typically 4-neighborhood).

### Energy-based formulation:

\[
P(X) \propto \exp\left(
\beta \sum_{(i,j)} \sum_{(k,l)\in \mathcal{N}(i,j)} X_{i,j} X_{k,l}
\right)
\]

### Parameters:

- \( \beta > 0 \): coupling strength
  - small β → noisy reconstruction
  - large β → overly smooth images

### Interpretation:

- If neighboring pixels match → higher probability
- Encourages connected strokes in digits

---

## 4. Observation Model (Noise Process)

We assume binary symmetric noise:

\[
P(Y_{i,j} = X_{i,j}) = 1 - \epsilon
\]
\[
P(Y_{i,j} \neq X_{i,j}) = \epsilon
\]

Equivalent:

\[
Y_{i,j} = X_{i,j} \oplus \eta_{i,j}, \quad \eta_{i,j} \sim \text{Bernoulli}(\epsilon)
\]

### Parameter:

- \( \epsilon \): corruption probability

---

## 5. Full Generative Model

The joint distribution:

\[
P(X, Y) = P(X) \prod_{i,j} P(Y_{i,j} \mid X_{i,j})
\]

Graph structure:

X(i,j) connected to:
   - X(i±1,j)
   - X(i,j±1)

X(i,j) → Y(i,j)

---

## 6. Inference Objective

We want:

### Posterior:

\[
P(X \mid Y)
\]

### Practical outputs:

- MAP estimate:
  \[
  \hat{X} = \arg\max_X P(X \mid Y)
  \]

- Marginals:
  \[
  P(X_{i,j}=1 \mid Y)
  \]

- Uncertainty maps

---

## 7. Why This is an Ising Model

The Ising model is:

\[
P(X) \propto \exp\left(\sum_i h_i X_i + \sum_{i,j} J_{ij} X_i X_j\right)
\]

Mapping:

- Pixels = spins
- \( J_{ij} = \beta \) for neighbors
- \( h_i = 0 \)

---

## 8. Infer.NET Structure

Variables:

- VariableArray2D<bool> X
- VariableArray2D<bool> Y

---

## 9. Spatial Coupling

Edges:
- (i,j) ↔ (i,j+1)
- (i,j) ↔ (i+1,j)

Soft equality constraints approximate Ising coupling.

---

## 10. Observation Model

For each pixel:

P(Y[i,j] | X[i,j]) = Bernoulli flip with ε

---

## 11. Inference

Infer.NET uses:
- Expectation Propagation (EP)
- Variational Message Passing (VMP)

Outputs:
- posterior marginals
- MAP reconstruction

---

## 12. Key Parameters

β = coupling strength  
ε = noise level  

---

## 13. Difficulties

- Loopy inference (no exact solution)
- Phase transitions in β
- Binary loss of grayscale info
- EP convergence issues
- Scaling limits on grid size

---

## 14. Design Options

1. Pure Ising MRF
2. Gaussian relaxation
3. Hybrid latent-logit model (recommended)

---

## 15. Extensions

- Class-conditioned β
- Learned β
- Anisotropic coupling
- Potts model extension

---

## 16. Relation to Modern Models

- CRFs
- diffusion models (conceptual link)
- denoising autoencoders

---

## 17. Summary

This is a:
- binary MRF
- spatial Bayesian model
- approximate inference system using Infer.NET

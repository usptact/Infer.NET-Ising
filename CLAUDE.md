# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is currently a **design specification** for a binary image denoising system using the Ising model and Infer.NET. The spec lives in `ising_mnist_infernet.md`. No implementation exists yet.

**Goal:** Denoise corrupted binary MNIST images (28×28, values in {0,1}) via Bayesian inference over a Markov Random Field.

## Model Architecture

The probabilistic model has two components:

1. **Ising prior** over latent clean image X — coupling strength β encourages neighboring pixels to agree:
   - `P(X) ∝ exp(β Σ_{neighbors} X_{i,j} X_{k,l})`
   - Small β → noisy, large β → over-smoothed

2. **Binary symmetric noise model** for observed image Y — each pixel flips independently with probability ε:
   - `P(Y_{i,j} | X_{i,j}) = Bernoulli flip with ε`

The graph structure connects each `X[i,j]` to its 4-neighbors and to the observed `Y[i,j]`.

## Infer.NET Implementation Plan

When implementing, use:
- `VariableArray2D<bool>` for both X (latent) and Y (observed)
- 4-neighborhood spatial coupling via soft equality constraints
- EP (Expectation Propagation) as the primary inference algorithm; VMP as fallback
- Outputs: posterior marginals `P(X_{i,j}=1 | Y)` and MAP reconstruction

Recommended design: **hybrid latent-logit model** (option 3 in §14 of the spec) over pure binary Ising or Gaussian relaxation.

## Known Difficulties

- Loopy inference — no exact solution on the 2D grid
- EP convergence issues under high β (near phase transition)
- Scaling: 28×28 = 784 variables with ~1500 edges; memory and runtime grow with grid size

## Technology Stack (intended)

- **.NET / C#** with [Infer.NET](https://dotnet.github.io/infer/) (Microsoft Research probabilistic programming framework)
- MNIST dataset for test images

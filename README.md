# nanoDiffusion

Minimal PyTorch implementations of:

* [DDPM (Denoising Diffusion Probabilistic Models)](https://arxiv.org/abs/2006.11239)
* [DDIM (Denoising Diffusion Implicit Models)](https://arxiv.org/abs/2010.02502)

with a lightweight DiT-based denoiser.

## Features

* Minimal and readable implementations
* DDPM training and sampling
* Deterministic DDIM sampling
* Equation-aligned code
* Pure PyTorch implementation
* Educational focus

## Repository Structure

```text
nanoDiffusion/
├── ddpm.py
├── ddim.py
└── model.py
```

## Theory and Derivations

Detailed explanations and mathematical derivations are available in the accompanying blog post:

* [Diffusion Models Notes](https://riteshrm.github.io/posts/diffusion-models/)

The blog covers:

* Forward diffusion process
* ELBO derivation
* Noise prediction objective
* DDPM sampling
* DDIM deterministic sampling

## Credits

The DiT backbone implementation in `model.py` is adapted from:

* https://github.com/sayakpaul/nanoDiT

The DDPM and DDIM implementations were written from scratch with a focus on minimalism and readability.


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

## Data

Download the dataset (Hugging Face) into `butterflies/`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="riteshrm/butterflies", repo_type="dataset", local_dir="butterflies"
)
```

The training scripts use `torchvision.datasets.ImageFolder`, so `butterflies/` should be laid out like:

```
butterflies/
  class_0/
  class_1/
  ...
```
## Usage

Train and Sample:

```bash
python ddpm.py
# or
python ddim.py
```

Note: the scripts currently assume `NUM_CLASSES = 5` and assert it matches the number of folders found under `butterflies/`.

During training, the script periodically:

- Saves sample grids as `sample_epoch_*.png`
- Saves checkpoints as `dit_conditional_epoch_*.pth`

## Training Progress

The GIFs below show generated samples transitioning from epoch 0 through the end of training.

### DDPM

![DDPM sample progression](ddpm.gif)

### DDIM

![DDIM sample progression](ddim.gif)

Most hyperparameters (image size, model size, batch size, number of steps, etc.) are defined at the top of each script.

## Credits

The DiT backbone implementation in `model.py` is adapted from:

* https://github.com/sayakpaul/nanoDiT

The DDPM and DDIM implementations were written from scratch with a focus on minimalism and readability.

# models
code for training a topographic model

* `config` contains model training and pre-optimization configs
* `init-data` contains data for pre-optimization
* `data` contains model training data (or infra for downloading)
* trained models are stored in `out`

## details
we use an almost-standard GPT-2 architecture (`model.py`), augmented with spatially embedded units (`positions.py`). unit positions are first pre-optimized against a set of ??? (`init_pos.py`), and then frozen. models are then trained to minimize a weighted combination of task and spatial loss (`train.py`). in other words, to train a model:

1. run `data/openwebtext/prepare.py` to prepare OWT for training
2. run `init_pos.py` (default args are in `config/init_pos.yaml`) to initialize positions
3. run `train.py` on your device with python (for CPU / a single GPU) or torchrun (for DDP with multiple GPUs)

code is built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) and [TDANN / spacetorch](https://github.com/neuroailab/TDANN) (margalit et al., 2023).

## pre optimization
we randomly select 50 square 'cortical neighborhoods' with radius 10. for each neighborhood, we iteratively perform 50 swapping perturbations, maintaining those that do not increase spatial loss. note this is far fewer swaps / neighborhoods than in the TDANN resnet implementation - this is because GPT layers are much smaller (just 28x28) such that 50 distinct 20x20 blocks cover effectively the entire layer.

## training
we minimize both task loss (cross-entropy on next-word prediction) and layer-wise spatial loss, defined as

$$\frac{1}{2} \cdot \left(1 - \mathrm{corr}(r, D)\right) \in [0, 1]$$

where $D$ is the inverse distance $D_i = 1 / (d_i + 1)$ and $r$ is the pairwise correlations of activations over a batch. computing these correlations is unwieldy for large `n_embed`, so on each forward pass, we randomly sample just 5 radius 10 neighborhoods to estimate the loss.

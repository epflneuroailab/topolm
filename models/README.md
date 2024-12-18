# models
code for training and finetuning a TDANN transformer LM

* `config` contains preset model training and pre-optimization configs
* `data` contains model training data / infra for downloading training data
* `gpt2-positions` contains saved position files
* trained models are stored in `out` and `finetuned`

## details
we use an almost-standard GPT-2 architecture (`model.py`), augmented with spatially embedded units (`positions.py`). unit positions are first randomly permuted (`init_pos.py`), and then frozen. models are then trained to minimize a weighted combination of task and spatial loss (`train.py`). in other words, to train a model:

1. run `data/fineweb/prepare.py` to prepare FineWeb-Edu 10B for training
2. run `init_pos.py` to initialize positions
3. run `train.py` on your device with python (for CPU / a single GPU) or torchrun (for DDP with multiple GPUs)

we trained each model on 4x NVIDIA 80GB A100s. default parameters are in `train_gpt2.yaml` and can be overridden on the command line, e.g.

```bash
torchrun --standalone --nproc_per_node=4 train.py train_gpt2.yaml batch_size=48 wandb_log=False
```

this section of the codebase is built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) and [TDANN / spacetorch](https://github.com/neuroailab/TDANN) (margalit et al., 2023).

## training
we minimize both task loss (cross-entropy on next-word prediction) and layer-wise spatial loss, defined as

$$\frac{1}{2} \cdot \left(1 - \mathrm{corr}(r, D)\right) \in [0, 1]$$

where $D$ is the inverse distance $D_i = 1 / (d_i + 1)$ and $r$ is the pairwise correlations of activations over a batch. computing these correlations is unwieldy for large `n_embed`, so on each forward pass, we randomly sample just 5 radius 10 neighborhoods to estimate the loss.

## position initialization

the standard position initialization scheme (and the one in the paper) is random permutation. however, we also experiment with 'pre-optimization' a la Margalit et al. (2024)'s CNNs. `init-data` contains data used for pre-optimization.

in pre-optimization (default args in `config/init_pos.yaml`), we randomly select 50 square 'cortical neighborhoods' with radius 10. for each neighborhood, we iteratively perform 50 swapping perturbations, maintaining those that do not increase spatial loss. note this is far fewer swaps / neighborhoods than in the TDANN resnet implementation - this is because GPT layers are much smaller (just 28x28) such that 50 distinct 20x20 blocks cover effectively the entire layer.
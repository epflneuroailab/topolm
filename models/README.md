# models
code for training a topographic model

* `config` contains model training and pre-optimization configs
* `init-data` contains data for pre-optimization
* `data` contains model training data (or infra for downloading)
* trained models are stored in `out`

## details
we use an almost-standard GPT-2 architecture (`model.py`), augmented with spatially embedded units (`positions.py`). unit positions are first pre-optimized against a set of ??? (`init_pos.py`), and then frozen. models are then trained to minimize a weighted combination of task and spatial loss (`train.py`).

code is built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) and [TDANN / spacetorch](https://github.com/neuroailab/TDANN) (margalit et al., 2023).

## pre optimization
we randomly select 10,000 circular 'cortical neighborhoods' with some radius. for each neighborhood, we iteratively perform 500 swapping perturbations, maintaining those that do not increase spatial loss.
# models

we (almost) use the standard GPT-2 architecture, augmented with spatially embedded units (for gpt-2, in 28x28 space; `positions.py`). unit positions are first pre-optimized (`init_pos.py`) against a set of ??? (`init-data`), and then frozen. models are then trained (`train.py`) to minimize a weighted combination of task and spatial loss.

code is built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) and [TDANN](https://github.com/neuroailab/TDANN) (margalit et al., 2023).

## pre optimization
at each layer, we randomly select 50 square 'cortical neighborhoods' with some radius. for each neighborhood, we iteratively perform 50 swapping perturbations, maintaining those that do not increase spatial loss. a radius of ~10 seems to have good pre-optimization performance (note that with a layer of size 28x28, there are only 64 20x20 neighborhoods).
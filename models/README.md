# models

we (almost) use the standard GPT-2 architecture, augmented with spatially embedded units (for gpt-2, in 28x28 space). unit positions are first pre-optimized against a set of ???, and then frozen. models are then trained to minimize a weighted combination of task and spatial loss.

code is built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) and [TDANN](https://github.com/neuroailab/TDANN) (margalit et al., 2023).

## pre optimization
we randomly select 10,000 circular 'cortical neighborhoods' with some radius. for each neighborhood, we iteratively perform 500 swapping perturbations, maintaining those that do not increase spatial loss.
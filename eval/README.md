# eval
scripts for evaluating topographic language models on neural benchmarks

## localizing the language network
* `extract.py` computes activations on the Fedorenko et al. (2010) sentence / non-word stimuli
* `localize.py` uses these activations to identify the core language network of the language model

after localizing the model's language-selective network, we can run response profile analyses:
* `profiles.py` computes response profiles on linguistic data across the whole network
* `by-cluster.py` applies a clustering algorithm to the language network and computes response profiles for each cluster

## part-of-speech clustering
* `responses.py` computes activations on a given set of stimuli (Moseley and Pulvermuller 2014 or Hauptman et al. 2024)
* `visualize.py` creates plots for these responses - both by condition and for specific pre-defined contrasts

activations / responses, once computed, are stored in `data`. all stimuli are stored in `stimuli`. visualizations are stored in `../figures/visualizations`.

## additional scripts
* `moran.py` computes Moran's I values for localized data (run this after `visualize.py`)
* `hrf.py` is a helper script that approximates fMRI readout sampling in the model

## other evaluations
we also include some scripts for running ablation studies (`ablate.py`) and computing raw activations on generic text input (`activations.py`).
# eval
scripts for evaluating topographic language models

* `extract` computes activations on the Fedorenko et al. (2010) sentence / non-word stimuli
* `localize` uses these activations to identify the core language network of the language model
* `responses` computes activations on a given set of stimuli (fedorenko 2024, moseley 2014, or elli 2019)
* `visualize` creates plots for these responses - both by condition and for specific pre-defined contrasts
* `profiles` computes mean activation profiles for all conditions of a stim set, both masked and unmasked

activations / responses, once computed, are stored in `data`. all stimuli are stored in `stimuli`. visualizations are stored in `../figures/visualizations`.
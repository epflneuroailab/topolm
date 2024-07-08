# topo-eval

evaluating topographic language models

* `extract.py`, `localize.py`, and `responses.py` are scripts for localization / response profiles
* `visualize.py` and `visualize-moseley.py` are scripts for visualizing activations on various inputs
* `cluster-regions.py` is a script for identifying neuron clusters
* `data` stores saved model activations
* `cramming` is a module for training / running topoformer-bert

## run localization
localization stimuli from [this link](https://www.dropbox.com/sh/c9jhmsy4l9ly2xx/AACQ41zipSZFj9mFbDfJJ9c4a?e=2&dl=0) should be loaded into the `fedorenko10_stimuli` folder.

to **extract** all activations on the Fedorenko stimuli, run (`sharing_strategy` is necessary for mac / windows)
```
python3 extract.py arch=bert-c5_topo impl.sharing_strategy=file_system
```

to **localize** these activations, run `localize.py`. to measure how the core language system responds to various stimuli, run
```
python3 responses.py arch=bert-c5_topo impl.sharing_strategy=file_system +stimuli $stim_name
```
with `$stim_name` in {`fedorenko`, `moseley`}.

## dependencies
basics:
* `torch`
* `R` / `tidyr`
* `tqdm`

the topoformer-bert code is written on top of the [cramming](https://github.com/JonasGeiping/cramming) repository, which requires
* huggingface `transformers` and `tokenizers`
* `hydra-core`
* `psutil`, `pynvml`, `safetensors`, `einops`
# topo-eval

evaluating topographic language models

## localization
run `extract.py`

## dependencies
basics:
* `torch`
* `R`
* `tqdm`

the topoformer-bert code is written on top of the [cramming](https://github.com/JonasGeiping/cramming) repository, which requires
* huggingface `transformers` and `tokenizers`
* `hydra-core`
* `psutil`, `pynvml`, `safetensors`, `einops`

localization stimuli from [this link](https://www.dropbox.com/sh/c9jhmsy4l9ly2xx/AACQ41zipSZFj9mFbDfJJ9c4a?e=2&dl=0) should be loaded into the `fedorenko10_stimuli` folder.
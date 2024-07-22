# topo-eval
training and evaluating topographic language models

## dependencies
you can install all python dependencies with `pip3 install -r requirements.txt`
* torch 2.x (for flash attn + compilation)
* huggingface datasets (for owt), tiktoken (for bpe)
* omegaconf, wandb, tqdm
* matplotlib, seaborn
* pandas, numpy, scipy

final analyses are done in R + tidyr

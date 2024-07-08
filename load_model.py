"""Example for a script to load a local saved model.

Use as e.g.

python load_local_model.py name=A6000amp_b4096_c5_o3_final base_dir=~/Documents/cmlscratch_backups/cramming/
> wandb=none impl.push_to_huggingface_hub=True arch=bert-c5 train=bert-o3 train.batch_size=4096
> data=c4-subset-processed dryrun=True +eval=GLUE_sane

"""

import os
import hydra

import cramming
from transformers import AutoTokenizer

MODEL_FILE = '~/projects/topo-eval/outputs/topotest/checkpoints/ScriptableMaskedLM_2024-07-03_9.9417/model.pth'

def main_load_process(cfg, setup):

    tokenizer = AutoTokenizer.from_pretrained("JonasGeiping/crammed-bert")
    model = cramming.construct_model(cfg.arch, tokenizer.vocab_size)
    
    model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.train, cfg.impl, setup=setup)
    model_path = os.path.expanduser(MODEL_FILE)
    model_engine.load_checkpoint(cfg.arch, model_path)

    model_engine.eval()

    # for name, layer in model.named_modules():
    #     print(name)
    for i in range(16):
        print('M:')
        print(model.encoder.layers[i].attn.self_attention.head.locality_mask.locality_weight)
        print('W_O:')
        print(model.encoder.layers[i].attn.dense.weight)

@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_load_process, job_name="load and push model")


if __name__ == "__main__":
    launch()
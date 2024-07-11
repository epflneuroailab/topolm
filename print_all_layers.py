"""
re-align topobert weights with new re-factored repository
"""

import os
import torch
import hydra

import cramming
from transformers import AutoTokenizer

def main_load_process(cfg, setup):

    tokenizer = AutoTokenizer.from_pretrained("JonasGeiping/crammed-bert")
    model = cramming.construct_model(cfg.arch, tokenizer.vocab_size)

    for layer in model.modules():
        print(layer)

@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_load_process, job_name="load and push model")

if __name__ == "__main__":
    launch()
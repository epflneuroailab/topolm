"""
re-align topobert weights with new re-factored repository
"""

import os
import torch
import hydra

import cramming
from transformers import AutoTokenizer

MODEL_FILE = '~/projects/topo-eval/outputs/topobert/checkpoints/ScriptableMaskedLM_2023-09-23_1.7895/model-old.pth'
FIXED_MODEL_FILE = '~/projects/topo-eval/outputs/topobert/checkpoints/ScriptableMaskedLM_2023-09-23_1.7895/model.pth'
# MODEL_FILE = '~/projects/topo-eval/outputs/topotest/checkpoints/ScriptableMaskedLM_2024-07-03_9.9417/model.pth'

def main_load_process(cfg, setup):

    tokenizer = AutoTokenizer.from_pretrained("JonasGeiping/crammed-bert")
    model = cramming.construct_model(cfg.arch, tokenizer.vocab_size)
    model_path = os.path.expanduser(MODEL_FILE)

    old_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_state_dict = model.state_dict()

    mapping_small = {
        'output_projection.weight' : 'fc_out.weight',
        'output_projection.bias' : 'fc_out.bias',
        'head.query.weight' : 'queries.weight',
        'head.key.weight' : 'keys.weight',
        'head.value.weight' : 'values.weight',
        'head.query.weight' : 'queries.weight'
    }

    locality_masks = [f'encoder.layers.{i}.attn.self_attention.head.locality_mask.locality_mask' for i in range(16)]
    biases = [f'encoder.layers.{i}.attn.self_attention.{qkv}.bias' for i in range(16) for qkv in ['queries', 'keys', 'values']]

    mapping = {}
    for i in range(16):
        for key in mapping_small:
            mapping[f'encoder.layers.{i}.attn.self_attention.' + mapping_small[key]] = f'encoder.layers.{i}.attn.self_attention.' + key

    new_state_dict = {}
    for old_key, value in old_state_dict.items():
        new_key = mapping.get(old_key, old_key)
        new_state_dict[new_key] = value

    for key in locality_masks:
        expected_shape = model_state_dict[key].shape
        new_state_dict[key] = torch.eye(expected_shape[0])

    model.load_state_dict(new_state_dict, strict = False)
    fixed_model_path = os.path.expanduser(FIXED_MODEL_FILE)
    torch.save(model.state_dict(), fixed_model_path)

    # new_state_dict = {}
    # for key in model.state_dict().keys():
    #     new_key = mapping.get(key, key)  # Use the new key if it exists, otherwise use the old key
    #     new_state_dict[new_key] = model.state_dict()[key]

    # torch.save(new_state_dict, fixed_model_path)

    # model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.train, cfg.impl, setup=setup)
    # model_engine.load_checkpoint(cfg.arch, fixed_model_path)

    # print(model_engine.state_dict().keys())

    # model_engine.eval()

    # for name, layer in model.named_modules():
    #     print(name)
    # for i in range(16):
    #     print('M:')
    #     print(model.encoder.layers[i].attn.self_attention.head.locality_mask.locality_weight)
    #     print('W_O:')
    #     print(model.encoder.layers[i].attn.dense.weight)

@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_load_process, job_name="load and push model")


if __name__ == "__main__":
    launch()
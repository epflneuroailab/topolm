import torch
import hydra
import logging
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cramming

log = logging.getLogger(__name__)

MODEL_FILE = '/home/tbh/Documents/cramming_local/outputs/amp_b4096_c5_o3_final_topobert/checkpoints/ScriptableMaskedLM_2023-09-23_1.7895/model.pth'
attention_dict = {"attention": []}

def hook_fn(module, input, output):
    attention_dict["attention"].append(output.mean(dim=1).detach().cpu())

def _register_hook(model, layer_name):
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer.register_forward_hook(hook_fn)

def apply_pca(attention_data, n_components=10, explained_variance_cutoff=5):
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(attention_data)
    components = pca.components_
    pca_variance = PCA(n_components=explained_variance_cutoff)
    pca_variance.fit(attention_data)
    variance_explained = pca_variance.explained_variance_ratio_
    return transformed_data, components, pca_variance.explained_variance_ratio_

def main_process(cfg, setup):
    cfg.impl['microbatch_size'] = 4

    layer_number = cfg.visualize.layer_number
    sublayer_type = cfg.visualize.sublayer_type

    layer_to_probe = f'encoder.layers.{layer_number}.attn.self_attention.head.{sublayer_type}'
    
    model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
    
    dataset, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl)
    model_engine, _, _, dataloader = cramming.load_backend(model, dataset, tokenizer, cfg.train, cfg.impl, setup=setup)
    
    model_engine.train(cfg.train.pretrain_in_train_mode)
    hook = _register_hook(model, layer_to_probe)
    model_engine.load_checkpoint(cfg.arch, MODEL_FILE)

    iterable_data = enumerate(dataloader)

    if cfg.train.gradinit.enabled:
        model_engine.gradinit(iterable_data, cfg.train.optim, cfg.train.gradinit)

    for step, batch in iterable_data:
        text = tokenizer.batch_decode(batch['input_ids'])
        device_batch = model_engine.to_device(batch)
        loss = model_engine.step(device_batch)

        if step == 100: # number of samples will be this time microbatch_size
            hook.remove()
            attention_data = torch.stack(attention_dict["attention"]).squeeze(1)
            attention_data = attention_data.reshape(-1, 784)
            print('Attention Data Shape:', attention_data.shape)
            pca, components, variance = apply_pca(attention_data)
            for idx, component in enumerate(components[:2]):
                plt.figure()
                sns.heatmap(component.reshape(28, 28), center=0, xticklabels=False, yticklabels=False, cbar=False)
                plt.title(f"{layer_to_probe}")
                plt.savefig(f"pca_weights_{idx}_{layer_to_probe}.png", format='png')
            break





@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_process, job_name="pretraining")


if __name__ == "__main__":
    launch()


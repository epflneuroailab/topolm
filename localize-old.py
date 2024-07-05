import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import scipy

from utils import read_pickle, write_pickle, get_layer_names

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('--model-name',  type=str,
                        default="gpt2", help='path of config file')
    parser.add_argument('--pretrained',  action='store_true',
                        help='use pretrained weights')
    parser.add_argument('--tokenizer-pretrained',  action='store_true',
                        help='use pretrained tokenizer')
    parser.add_argument('--threshold',  type=float,
                        default=0.05, help='p-value threshold')
    parser.add_argument('--seed',  type=int,
                        default=42, help='seed')
    parser.add_argument('--num-units',  type=int,
                        default=None, help='take top num units')
    parser.add_argument('--embed-agg',  type=str,
                        default="last-token", help='embedding aggregation')
    parser.add_argument('--num-attn-heads',  type=int,
                        default=32, help='number of attention heads')
    parser.add_argument('--num-blocks',  type=int,
                        default=1, help='number of blocks')
    parser.add_argument('--num-cycles',  type=str,
                        default=1, help='number of cycles')
    parser.add_argument('--init-range',  type=float,
                        default=0.02, help='initialization range') 
    args = parser.parse_args()

    model_name_ = args.model_name.split("/")[-1]

    # if args.pretrained a:
    #     path = f"dumps/layer_representations_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_nsamples=240.pkl"
    # else:
    # path = f"dumps/layer_representations_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_nsamples=240_seed={args.seed}_nheads={args.num_attn_heads}_init-range={args.init_range}.pkl"
    
    if args.pretrained:
        if model_name_ in ["gpt2-xl"]:
            path = f"dumps/layer_representations_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_nsamples=240_seed={args.seed}_nheads={args.num_attn_heads}_init-range={args.init_range}.pkl"
        else:
            path = f"dumps/layer_representations_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_nsamples=240_seed={args.seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={args.num_cycles}_init-range={args.init_range}.pkl"
    else:
        path = f"dumps/layer_representations_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_nsamples=240_seed={args.seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={args.num_cycles}_init-range={args.init_range}.pkl"

    path = f"dumps/reps_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_seed={args.seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={args.num_cycles}_init-range={args.init_range}_tok={args.tokenizer_pretrained}.pkl"
    # save_path = f"dumps/language-mask_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_nunits={args.num_units}_seed={args.seed}_nheads={args.num_attn_heads}_init-range={args.init_range}.pkl"
    save_path = f"dumps/l-mask_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_nunits={args.num_units}_seed={args.seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={args.num_cycles}_init-range={args.init_range}_tok={args.tokenizer_pretrained}_v2.pkl"
    # if os.path.exists(save_path):
    #     print(f"> Already Exists: {save_path}")
    #     exit()

    data = read_pickle(path)

    layer_names = get_layer_names(model_name_, None)
    print(layer_names)

    num_units = data["sentences"][layer_names[0]].shape[-1]

    language_mask = np.zeros((len(layer_names), num_units))
    language_prob_mask = np.zeros((len(layer_names), num_units))

    p_values_matrix = np.zeros((len(layer_names), num_units))
    t_values_matrix = np.zeros((len(layer_names), num_units))

    for layer_idx, layer_name in enumerate(tqdm(layer_names)):
        if layer_name == "transformer.wte":
            continue 

        sentences_actv = data["sentences"][layer_name]
        non_words_actv = data["non-words"][layer_name]

        t_values_matrix[layer_idx], p_values_matrix[layer_idx] = scipy.stats.ttest_ind(sentences_actv, non_words_actv, axis=0, equal_var=False)


    if args.num_units is not None:
        language_mask = is_topk(t_values_matrix, k=args.num_units)
        language_prob_mask = None
        save_path = f"figures/l-mask_model={model_name_}_dataset=fed10_pretrained={args.pretrained}_agg={args.embed_agg}_nunits={args.num_units}_seed={args.seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={args.num_cycles}_init-range={args.init_range}_tok={args.tokenizer_pretrained}_v2.png"
    else:
        adjusted_p_values = scipy.stats.false_discovery_control(p_values_matrix.flatten())
        adjusted_p_values = adjusted_p_values.reshape((len(layer_names), num_units))
        language_mask = (adjusted_p_values < args.threshold) & (t_values_matrix > 0) 
        language_prob_mask = 1 - (adjusted_p_values * (t_values_matrix > 0))
        save_path = f"figures/language-mask_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_thresh={args.threshold}_seed={args.seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={args.num_cycles}_init-range={args.init_range}.png"

    num_active_units = language_mask.sum()
    total_num_units = np.prod(language_mask.shape)

    if args.num_units is not None:
        desc = f"# of Active Units: {num_active_units:,}/{total_num_units:,} = {(num_active_units/total_num_units)*100:.2f}%"
    else:
        desc = f"# of Active Units: {num_active_units:,}/{total_num_units:,} = {(num_active_units/total_num_units)*100:.2f}% | {args.threshold}"
        
    print(desc)

    sns.heatmap(language_mask)
    plt.title(desc)
    plt.savefig(save_path)
    
    plt.clf()
    plt.cla()
    plt.close()

    if language_prob_mask is not None:
        sns.heatmap(language_prob_mask, vmin=1-args.threshold)
        plt.title(desc)
        plt.savefig(f"figures/language-prob-mask_model={model_name_}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_thresh={args.threshold}_nheads={args.num_attn_heads}_init-range={args.init_range}.png")
    
    savepath = f"dumps/{os.path.basename(save_path)[:-4]}.pkl"
    print(f"> Saving @ {savepath}")
    write_pickle(savepath, language_mask)


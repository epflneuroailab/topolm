import numpy as np
import pandas as pd
import pickle as pkl
import scipy

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import libpysal as lp
from esda.moran import Moran

cfg = OmegaConf.from_cli()
model_name = cfg.name

results = defaultdict(list)
conditions = [
    ('moseley', 'concrete-concrete'),
    ('moseley', 'abstract-concrete'),
    ('moseley', 'abstract-abstract'),
    ('moseley', 'noun-verb'),
    ('elli',    'noun_verb')
]

for smoothing in ['smoothed', 'unsmoothed']:
    print(smoothing + '...')
    
    for condition in conditions:
        
        with open(f'data/contrasts/{model_name}/{condition[0]}-{smoothing}/{condition[1]}.pkl', 'rb') as f:
            data = pkl.load(f)
        
        for layer_idx in range(12):
            w = lp.weights.lat2W(28, 28, rook=False)
            moran = Moran(data['t_values'][layer_idx].flatten(), w)
            results['_'.join(condition)].append(moran.I)
    
    df = pd.DataFrame(results)
    df.to_csv(f'moran/{model_name}_{smoothing}.csv')
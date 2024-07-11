import numpy as np
import numpy.linalg as LA

import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt

MASK_FILE = 'data/topobert/lmask.pkl'
SAVEPATH = 'data/topobert/regions.pkl'

def kmeans(X, k):
    # randomly select k centers
    Z = X[np.random.choice(X.shape[0], size=3, replace=False), :]
    C = get_choices(X, Z)

    for _ in range(1000):
        Z = [(1/len(c)) * sum(c) for c in C]
        C = get_choices(X, Z)

    return Z, [np.array(y) for y in C]

def get_choices(X, Z):
    C = [[] for _ in Z]

    for x in X:
        scores = [LA.norm(z - x) ** 2 for z in Z]
        C[scores.index(min(scores))].append(x)

    return C

if __name__ == "__main__":

    with open(MASK_FILE, 'rb') as f:
        mask = pkl.load(f)

    layer_names = [f'encoder.layers.{i}.attn.dense' for i in range(16)]

    coordinates = [[] for _ in range(16)]
    k_means_results = {layer_name : {'centers' : [], 'coordinates' : []} for layer_name in layer_names}

    for i in range(mask.shape[0]):

        coordinates[i] = np.argwhere(mask[i].reshape(28, 28) == 1)

        Z, C = kmeans(coordinates[i], 3)
        
        k_means_results[layer_names[i]]['centers'] = Z
        k_means_results[layer_names[i]]['coordinates'] = C

    with open(SAVEPATH, 'wb') as f:
        pkl.dump(k_means_results, f)

    # print(coordinates[0])

    # print(C[0].shape)

    # plt.scatter(Z[:, 0], Z[:, 1], color = 'purple')
    # plt.scatter(C[0][:, 0], C[0][:, 1], color = 'r')
    # plt.scatter(C[1][:, 0], C[1][:, 1], color = 'g')
    # plt.scatter(C[2][:, 0], C[2][:, 1], color = 'b')
    # plt.show()

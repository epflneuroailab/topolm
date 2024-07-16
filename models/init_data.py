import os
import tiktoken
import numpy as np

with open('init/jabberwocky.txt', 'r', encoding='utf-8') as f:
    data = f.read()

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(data)

train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'init/train.bin'))
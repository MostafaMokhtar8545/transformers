import torch
import torch.nn as nn
import torch.nn.functional as F

# download dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Read input
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoding and decoding functions
itos = {i: s for i, s in enumerate(chars)}
stoi = {s: i for i, s in enumerate(chars)}
def encode(s):
    return [stoi[l] for l in s]

def decode(li):
    return ''.join(itos[i] for i in li)


# Train/Dev split
data = torch.tensor(encode(text), dtype=int)
n = round(.9 * len(data))
train_data = data[:n]
val_data = data[n:]


block_size = 8
batch_size = 32
n_embd = 32
# Generate batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    Xb = torch.stack([data[i:i+block_size] for i in ix])
    Yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return Xb, Yb



class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.tranpose(-2, -1) * C**-.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, -1)
        out = wei @ v
        return out
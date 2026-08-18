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
# Generate batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    Xb = torch.stack([data[i:i+block_size] for i in ix])
    Yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return Xb, Yb


# bigram : predict next token based on only the current one.
class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, target=None):
        x = self.emb(x)
        if target == None:
            loss = None
        else:
            B, T, C = x.shape
            x = x.view(-1, C)
            target = target.view(-1)
            loss = F.cross_entropy(x, target)
            x = x.view(B, T, C)
        return x, loss

    def generate(self, ix, max_new_tokens):
        logits, loss = self(ix)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        # multinomial takes 1D or 2D tensors only
        # always sample from the last dimension.
        # (N,) -> (num_samples,)
        #(B, N) -> (B, num_samples)
        idx_next = torch.multinomial(probs, num_samples=1) 
        idx = torch.cat((ix, idx_next), dim=1)
        return idx


model = Bigram()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

#Training
for steps  in range(20000):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if(steps == 0):
        print(f'start loss {loss.item()}')

print(f'final loss: {loss.item()}')

# Saving the model
torch.save(model.state_dict(), 'bigram.pth')
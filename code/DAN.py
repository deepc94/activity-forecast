import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

np.random.seed(1)
torch.manual_seed(1)
torch.set_default_dtype(torch.float64)

class DAN(nn.Module):

    def __init__(self, d_emb=16, hidden_sizes=[512, 256, 128], dropout=0.25):
        super(DAN, self).__init__()
        self.d_emb = d_emb
        self.dropout = nn.Dropout(p=dropout)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=d_emb) # o/p: (bsz, seq, 5, 16) TODO:flatten last 2
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(in_features=4*225+5*d_emb, out_features=hidden_sizes[0])) #
        for k in range(len(hidden_sizes)-1):
            self.hidden.append(nn.Linear(in_features=hidden_sizes[k], out_features=hidden_sizes[k+1]))
        self.out = nn.Linear(in_features=hidden_sizes[-1], out_features=5)

    def forward(self, X, X_y, Y_x):
        bsz, seq_len = X.size()[:2]
        X_y = self.emb(X_y).view(bsz, seq_len, 5*self.d_emb) # (bsz, seq_len, 80)
        X = torch.cat((X, X_y), dim=-1) # (bsz, seq_len, 80+675)
        X = torch.cat((X.mean(dim=1), Y_x), dim=-1)
        for layer in self.hidden:
            X = self.dropout(torch.relu(layer(X)))
        preds = torch.sigmoid(self.out(X))
        return preds

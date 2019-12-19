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





class APM(nn.Module):

    def __init__(self, d_emb=16, d_hid=512, d_lab=32, n_layers=1, dropout=0.0, bi=False):
        super(APM, self).__init__()
        self.d_emb = d_emb
        self.d_hid = d_hid
        self.d_lab = d_lab
        self.n_layers = n_layers
        self.n_dir = 2 if bi else 1
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=d_emb) # o/p: (bsz, seq, 5, 16) TODO:flatten last 2
#         self.label_enc = nn.GRU(input_size=d_emb*5, hidden_size=d_lab,
#                         num_layers=n_layers, batch_first=True, dropout=dropout,
#                         bidirectional=bi)
        self.enc = nn.GRU(input_size=225*3+d_emb*5, hidden_size=d_hid,
                        num_layers=n_layers, batch_first=True, dropout=dropout,
                        bidirectional=bi) # o/p: output (bsz, seq_len, n_dir * d_hid)  input: 
        self.dense = nn.Linear(in_features=n_layers*self.n_dir*(d_hid)+225, out_features=32) #(d_hid+d_lab)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(in_features=32, out_features=5) #TODO: stack h_n with y_x
        # self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, X, X_y, Y_x):
        bsz, seq_len = X.size()[:2]
        X_y = self.emb(X_y).view(bsz, seq_len, 5*self.d_emb) # (bsz, seq_len, 80)
#         _, h_n_y = self.label_enc(X_y) # (n_layers * n_dir, bsz, d_lab)
        X = torch.cat((X, X_y), dim=-1) # (bsz, seq_len, 80+675)
        _, h_n = self.enc(X) # (n_layers * n_dir, bsz, d_hid)
#         h_n = torch.cat((h_n, h_n_y), dim=-1)

        Y_x = torch.cat((h_n.transpose(0,1).contiguous().view(bsz, -1), Y_x), dim=-1) # (bsz, 225+n_layers*(d_hid+d_lab))
        Y_x = torch.relu(self.dense(Y_x))
        preds = torch.sigmoid(self.out(Y_x)) # (bsz, 5)
        return preds

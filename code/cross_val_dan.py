# coding: utf-8

import numpy as np
np.random.seed(1)
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(1)
torch.set_default_dtype(torch.float64)
dtype = torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = torch.cuda.device_count()*4
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from code.DAN import DAN

def plot_line(filepath, X,Y,title,xlabel,ylabel,ticks=False):
    # function to make line plot for given X,Y
    plt.plot(X, Y, '-bo')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ticks:
        plt.xticks(())
        plt.yticks(())
    plt.savefig(filepath)

def fit(loader_train, loader_val, model, optimizer, epochs=5):

    for e in range(epochs):
        for t, (x, x_y, y_x, y) in enumerate(loader_train):
            model.train()
            
            x = x.to(device=device, dtype=dtype)
            x_y = x_y.to(device=device, dtype=torch.long)
            y_x = y_x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

            preds = model(x, x_y, y_x)
            loss = F.binary_cross_entropy(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_fscore = f_score(loader_val, model)
        print('Epoch {0}, loss = {1:.4f}, avg_val_fscore = {2:.4f}'.format(e, loss.item(), val_fscore))
    final_val_fscore = f_score(loader_val, model)
    return (model, final_val_fscore)

def f_score(loader_val, model, average='micro'):

    model.eval()  # set model to evaluation mode
    true = []
    pred = []
    with torch.no_grad():
        for x, x_y, y_x, y in loader_val:
            x = x.to(device=device, dtype=dtype)
            x_y = x_y.to(device=device, dtype=torch.long)
            y_x = y_x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            preds = model(x, x_y, y_x)
            preds = preds > 0.5
            true.append(y.cpu().numpy())
            pred.append(preds.cpu().numpy())
    
    true = np.concatenate(true, axis=0)
    pred = np.concatenate(pred, axis=0)

    f_scores = []
    for i in range(5):
        f_scores.append(f1_score(true[:,i], pred[:,i], average=average))
    print('Per class f1_score: ', f_scores)
    return (np.mean(f_scores))


def main():
    train = np.load('data/train_rand60.npz')
    val = np.load('data/val_rand60.npz')
    tr_X, tr_X_y, tr_Y, tr_Y_x = train['tr_X'], train['tr_X_y'], train['tr_Y'], train['tr_Y_x']
    val_X, val_X_y, val_Y, val_Y_x = val['val_X'], val['val_X_y'], val['val_Y'], val['val_Y_x']

    train_data = TensorDataset(torch.from_numpy(tr_X[:,:,:]).double(),
                              torch.from_numpy(tr_X_y[:,:,:]).long(),
                              torch.from_numpy(tr_Y_x[:,:]).double(),
                              torch.from_numpy(tr_Y[:,:]).long())

    val_data = TensorDataset(torch.from_numpy(val_X[:,:,:]).double(),
                              torch.from_numpy(val_X_y[:,:,:]).long(),
                              torch.from_numpy(val_Y_x[:,:]).double(),
                              torch.from_numpy(val_Y[:,:]).long())

    loader_train = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=num_workers)
    loader_val = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=num_workers)

    # hyperparameter search
    lrs = [0.0001, 0.001, 0.01]
    d_hids = [[512, 128], [256, 32], [512, 256, 32], [256, 128, 32]]
    d_embs = [16, 32]
   
    best_val = 0.0
    
    val = []
    best_hyperparam = {'lr':None, 'd_emb':None, 'd_hid':None}
    for d_emb in tqdm(d_embs, desc='Embedding dim: '):
        for d_hid in tqdm(d_hids, desc='Hidden Size: '):
            for lr in tqdm(lrs, desc='Learning rate: '):
                print("***** Running model for lr={},  d_hid={}, d_emb={} *****".format(lr,  d_hid, d_emb))
                hyperparam = {'lr':lr, 'd_hid':d_hid, 'd_emb':d_emb}
                model = DAN(d_emb=d_emb, hidden_sizes=d_hid, dropout=0.25)
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(device=device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
                model, val_fscore = fit(loader_train, loader_val, model, optimizer, epochs=10)
                val.append(val_fscore)
                combo = 'lr={}_d_hid={}_d_emb={}'.format(lr, d_hid, d_emb)
                if val_fscore > best_val: 
                    best_val=val_fscore
                    best_hyperparam.update(hyperparam)
                    model_name = 'models/best_DAN_{}.pth'.format(combo)
                    torch.save(model.state_dict(), model_name)
                print()
    print('best hyperparam found at: ', best_hyperparam)
    print('best validation fscore obtained = {}'.format(best_val))
    plot_line('dan_cv.png', np.arange(len(val)), val, 'DAN cross validation', 'hyperparam combo', 'f1-score')

if __name__ == '__main__':
    main()





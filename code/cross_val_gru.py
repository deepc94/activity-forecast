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
from code.GRU import APM

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
    best_val = 0.0
    ep = None
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
        if val_fscore > best_val:
            best_val = val_fscore
            ep = e
        print('Epoch {0}, loss = {1:.4f}, avg_val_fscore = {2:.4f}'.format(e, loss.item(), val_fscore))
    print()
    print('Best val fscore obtained at epoch {}, fscore={}'.format(ep, best_val))
    return best_val

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
    
    subset_idx = np.random.choice(tr_Y.shape[0], 25024)
    train_data = TensorDataset(torch.from_numpy(tr_X[subset_idx,:,:]).double(),
                              torch.from_numpy(tr_X_y[subset_idx,:,:]).long(),
                              torch.from_numpy(tr_Y_x[subset_idx,:]).double(),
                              torch.from_numpy(tr_Y[subset_idx,:]).long())

    val_data = TensorDataset(torch.from_numpy(val_X[:,:,:]).double(),
                              torch.from_numpy(val_X_y[:,:,:]).long(),
                              torch.from_numpy(val_Y_x[:,:]).double(),
                              torch.from_numpy(val_Y[:,:]).long())

    loader_train = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    loader_val = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)
    
    # hyperparameter search
    lrs = [0.0001, 0.001, 0.01]
    d_hids = [64, 128, 256, 512]
    n_layers = [1,2]
   
    best_val = 0.0
    
    val = []
    best_hyperparam = {'lr':None, 'd_hid':None, 'n_layer':None}
    for n_layer in tqdm(n_layers, desc='No. of layers: '):
        for d_hid in tqdm(d_hids, desc='Hidden Size: '):
            for lr in tqdm(lrs, desc='Learning rate: '):
                print("***** Running model for lr={}, d_hid={}, n_layer={} *****".format(lr,  d_hid, n_layer))
                
                hyperparam = {'lr':lr, 'd_hid':d_hid, 'n_layer':n_layer}
                if n_layer == 1:
                    model = APM(d_emb=16, d_hid=d_hid, n_layers=n_layer, dropout=0)
                else:
                    model = APM(d_emb=16, d_hid=d_hid, n_layers=n_layer, dropout=0.25)

                model = model.to(device=device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
                val_fscore = fit(loader_train, loader_val, model, optimizer, epochs=5)
                val.append(val_fscore)
                combo = 'lr={}_d_hid={}_nlayers={}'.format(lr, d_hid, n_layer)
                if val_fscore > best_val: 
                    best_val=val_fscore
                    best_hyperparam.update(hyperparam)
                    model_name = 'models/best_GRU_{}.pth'.format(combo)
                    torch.save(model.state_dict(), model_name)
                print()
    print('best hyperparam found at: ', best_hyperparam)
    print('best validation fscore obtained = {}'.format(best_val))
    plot_line('dan_gru.png', np.arange(len(val)), val, 'Simple GRU cross validation', 'hyperparam combo', 'f1-score')

if __name__ == '__main__':
    main()





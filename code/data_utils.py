import numpy as np
import pandas as pd
import random
random.seed(1)
import torch
from torch.utils.data import Sampler, Dataset
from collections import OrderedDict

def to_numpy(ts, xx, yy, msk, scaler):
    """Function to convert multi-user dataframes to ndarrays"""
    t = []
    x = []
    y = []
    m = []
    
    for i,_ in ts.groupby(level=0):
        # timestamps
        t.append(ts.loc[i].values)
        # features
        x.append(scaler.transform(xx.loc[i].values))
        # labels
        sample_df = yy.loc[i].loc[:,'label:LYING_DOWN':'label:OR_standing']
        sample_df.interpolate(method='nearest', axis=0, limit_direction='both', inplace=True)
        sample_df.fillna(method='ffill', axis=0, inplace=True)
        sample_df.fillna(method='bfill', axis=0, inplace=True)
        sample_df.fillna(0., axis=0, inplace=True)
        y.append(sample_df.values)
        # masks
        m.append(msk.loc[i].values)
        
    return (t, x, y, m)


def create_examples(timestamps, Xs, ys, masks, seq_len=30):
    """Function to divide original data into data cases for training"""
    X = []
    X_y = []
    Y = []
    Y_x = []
    for user in range(len(Xs)): #, desc='User: '):
        print("Processing User {}".format(user+1))
        ts_user = timestamps[user]
        X_user = Xs[user]
        y_user = ys[user]
        msk_user = masks[user]
        n_train = 0
        for i in range(X_user.shape[0]): #, desc='Batch: '):
            end_ix = i+np.random.choice(np.arange(1,seq_len+1))
#             end_ix = i+seq_len
            if end_ix > Xs[user].shape[0]-1:
                break
            xx = X_user[i:end_ix, :] # input features
            msk = msk_user[i:end_ix, :] # input masks
            msk_comp = np.where(msk==0, 1, -1)
            xy = y_user[i:end_ix] # input labels
            
            # computation for delta_t
            ts = ts_user[i:end_ix]
            S = np.ones_like(msk) * ts.reshape(-1,1)
            delta_t = np.zeros_like(msk) # input delta times
            for t in range(1,delta_t.shape[0]):
                delta_t[t,:] = S[t,:] - S[t-1,:] + msk[t-1,:]*delta_t[t-1,:]
            
            # find the last target within 1 hr from now
            y_ix = end_ix
            last_ts = ts_user[end_ix-1]
            choose_y_ix = []
            while (y_ix < Xs[user].shape[0]) and (ts_user[y_ix]-last_ts <= 3600):
                choose_y_ix.append(y_ix)
                y_ix += 1
                
            if (y_ix == end_ix): # no targets within 1 hr
                continue
#             y = y_user[end_ix:y_ix]
#             yx = np.ones(msk.shape[1])*ts_user[end_ix:y_ix][:,None] - S[-1,:] + msk[-1,:]*delta_t[-1,:]  
            y_ix = np.random.choice(choose_y_ix) # pick a random target within 1 hr
            y = y_user[y_ix] # target label
            yx = np.ones(msk.shape[1])*ts_user[y_ix] - S[-1,:] + msk[-1,:]*delta_t[-1,:] # target delta time
            Y.append(y.astype(np.int64))
            Y_x.append(yx/3600.)
            X.append(np.hstack((xx, msk_comp, delta_t/3600.)))
            X_y.append(xy.astype(np.int64))
                
            n_train += 1
            
        print("Added {} examples from User {}".format(n_train, user+1))

    return (X, X_y, np.array(Y), np.array(Y_x))

class ActivityDataset(Dataset):
    """Custom dataset for variable length inputs"""
    def __init__(self, X, X_y, Y_x, Y):
        self.X = X # (75k, 1-30, 675)
        self.X_y = X_y # (75k, 1-30, 5)
        self.Y_x = Y_x # (75k, 675)
        self.Y = Y # (75k, 5)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return (torch.from_numpy(self.X[index]).double(), 
                torch.from_numpy(self.X_y[index]).long(), 
                torch.from_numpy(self.Y_x[index]).double(), 
                torch.from_numpy(self.Y[index]).long())


class BucketBatchSampler(Sampler):
    """Custom sampler to batch equal length inputs"""
    def __init__(self, X_y, train=True, batch_size=64):
        self.batch_size = batch_size
        self.train = train
        ind_n_len = []
        for i, xy in enumerate(X_y):
            ind_n_len.append((i, xy.shape[0]))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        if self.train:
            random.shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        if self.train:
            random.shuffle(self.batch_list)
        for ix in self.batch_list:
            yield ix
            
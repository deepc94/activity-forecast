import sys
sys.path.append("code/")
sys.path.append("data/")
sys.path.append("models/")

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
from data_utils import to_numpy, create_examples, ActivityDataset, BucketBatchSampler
import pickle
from tqdm import tqdm
np.random.seed(1)
torch.manual_seed(1)
torch.set_default_dtype(torch.float64)
dtype = torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from GRU import APM
from DAN import DAN


class activity_prediction_model:
    """Activity prediction
    
        You may add extra keyword arguments for any function, but they must have default values 
        set so the code is callable with no arguments specified.
    
    """
    def __init__(self, ):
        self.model = APM(d_emb=32, d_hid=128, n_layers=2, dropout=0.25)
        # self.model = DAN(d_emb=32, hidden_sizes=[512, 128], dropout=0.25)
    
    def fit(self, df):
        """Train the model using the given Pandas dataframe df as input. The dataframe
        has a hierarchical index where the outer index (ID) is over individuals,
        and the inner index (Time) is over time points. Many features are available.
        There are five binary label columns: 
        
        ['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking', 'label:TALKING', 'label:OR_standing']

        The dataframe contains both missing feature values and missing label values
        indicated by nans. Your goal is to design and fit a probabilistic forecasting 
        model that when given a dataframe containing a sequence of incomplete observations 
        and a time stamp t, outputs the probability that each label is active (e.g., 
        equal to 1) at time t.

        Arguments:
            df: A Pandas data frame containing the feature and label data
        """
        
        ts_train, X_train, y_train = df.iloc[:,0], df.iloc[:,1:226], df.iloc[:,226:]
        missing_mask_train = X_train.isna().astype(int)
        scaler = MinMaxScaler((-1,1))
        scaler.fit(X_train.values)
        joblib.dump(scaler, 'data/scaler.pkl') 
        with open('data/mean.pkl', 'wb') as f:
            pickle.dump(X_train.mean(), f)

        X_train.fillna(X_train.mean(), inplace=True)
        ts_tr, X_tr, y_tr, m_tr = to_numpy(ts_train, X_train, y_train, missing_mask_train, scaler)
        tr_X, tr_X_y, tr_Y, tr_Y_x = create_examples(ts_tr, X_tr, y_tr, m_tr)

        train_batch_sampler = BucketBatchSampler(tr_X_y, train=True, batch_size=64)
        train_dataset = ActivityDataset(tr_X, tr_X_y, tr_Y_x, tr_Y)
        loader_train = DataLoader(train_dataset, batch_size=1, batch_sampler=train_batch_sampler, shuffle=False, num_workers=4, drop_last=False)

        self.model = self.model.to(device=device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

        epochs=5
        for e in tqdm(range(epochs), desc='Epochs: '):
            for t, (x, x_y, y_x, y) in enumerate(tqdm(loader_train, desc='Iterations: ', leave=True)):
                self.model.train()
                
                x = x.to(device=device, dtype=dtype)
                x_y = x_y.to(device=device, dtype=torch.long)
                y_x = y_x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=dtype)

                preds = self.model(x, x_y, y_x)
                loss = F.binary_cross_entropy(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print('Epoch {0}, loss = {1:.4f}'.format(e, loss.item()))

        self.model = self.model.to(device=torch.device('cpu'))

        
    def forecast(self, df, t):
        """Given the feature data and labels in the dataframe df, output the log probability
        that each labels is active (e.g., equal to 1) at time t. Note that df may contain
        missing label and/or feature values. Assume that the rows in df are in time order, 
        and that all rows are for data before time t for a single individual. Any number of 
        rows of data may be provided as input, including just one row. Further, the gaps
        between timestamps for successive rows may not be uniform. t can also be any time 
        after the last observation in df. There are five labels to predict:
        
        ['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking', 'label:TALKING', 'label:OR_standing']

        Arguments:
            df: a Pandas data frame containing the feature and label data for multiple time 
            points before time t for a single individual.
            t: a unix timestamp indicating the time to issue a forecast for

        Returns:
            pred: a python dictionary containing the predicted log probability that each label is
            active (e.g., equal to 1) at time t. The keys used in the dictionary are the label 
            column names listed above. The values are the corresponding log probabilities.
        
        """
        with open('data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('data/mean.pkl', 'rb') as f:
            x_train_mean = pickle.load(f)

        ts_val, X_val, y_val = df.iloc[:,0], df.iloc[:,1:226], df.iloc[:,226:]
        missing_mask_val = X_val.isna().astype(int)
        X_val.fillna(x_train_mean, inplace=True)
        ts_va, X_va, y_va, m_va = to_numpy(ts_val, X_val, y_val, missing_mask_val, scaler)

        ts, xx, xy, msk = ts_va[0], X_va[0], y_va[0], m_va[0]
        msk_comp = np.where(msk==0, 1, -1)
        # computation for delta_t
        S = np.ones_like(msk) * ts.reshape(-1,1)
        delta_t = np.zeros_like(msk) # input delta times
        for t_step in range(1,delta_t.shape[0]):
            delta_t[t_step,:] = S[t_step,:] - S[t_step-1,:] + msk[t_step-1,:]*delta_t[t_step-1,:]

        yx = np.ones(msk.shape[1])*t - S[-1,:] + msk[-1,:]*delta_t[-1,:] 
        X = np.hstack((xx, msk_comp, delta_t/3600.))
        X_y = xy.astype(np.int64)
        Y_x = yx/3600.

        self.model.eval()
        pred = self.model.forward(torch.from_numpy(X[None,:]).double(), 
                                torch.from_numpy(X_y[None,:]).long(), 
                                torch.from_numpy(Y_x[None,:]).double())

        pred = np.log(pred.detach().numpy().squeeze())
        return {'label:LYING_DOWN':pred[0], 'label:SITTING':pred[1], 'label:FIX_walking':pred[2], 'label:TALKING':pred[3], 'label:OR_standing':pred[4]}
        
        
    def save_model(self):
        """A function that saves the parameters for your model. You may save your model parameters 
           in any format. You must upload your code and your saved model parameters to Gradescope.
           Your model must be loadable using the load_model() function on Gradescope. Note:
           your code will be loaded as a module from the parent directory of the code directory using: 
           from code.activity_prediction import activity_prediction_model. You need to take this into
           account when developing a method to save/load your model. 

        Arguments:
            None
        """
        
        torch.save(self.model.state_dict(), 'models/best_model_gru.pth')


    def load_model(self):
        """A function that loads parameters for your model, as created using save_model(). 
           You may save your model parameters in any format. You must upload your code and your 
           saved model parameters to Gradescope. Following a call to load_model(), forecast() 
           must also be runnable with the loaded parameters. Note: your code will be loaded as 
           a module from the parent directory of the code directory using: 
           from code.activity_prediction import activity_prediction_model. You need to take this into
           account when developing a method to save/load your model 

        Arguments:
            None
        """
        
        self.model.load_state_dict(torch.load('models/best_model_gru.pth'))


def main():
    
    #Load the training data set
    df=pd.read_pickle("data/train_data.pkl", compression='gzip')
    df_train = df.loc[list(range(11))+list(range(12,21))]
    df_val = df.loc[list(range(23,26))]
    #Create the model
    apm = activity_prediction_model()
    
    #Fit the model
    # apm.fit(df_train)
    
    #Save the model
    # apm.save_model()
    
    #Load the model
    apm.load_model()
    
    #Get a sample of data
    example = df_val.loc[[23]][:10]
    print(example["timestamp"][-1])
    #Get a timestamp 5 minutes past the end of the example
    t = example["timestamp"][-1] + 5*60 

    
    #Compute a forecast
    f = apm.forecast(example, t)
    print(f)
    
if __name__ == '__main__':
    main()
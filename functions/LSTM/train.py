import numpy as np
import pandas as pd
import torch
import os
import math 
import torch.nn as nn

from functions.misc import *

#####################################################
#####################################################

def work(df, df_train, data_loader, model, loss_function, scalers, optimizer, settings):

    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    position = 0
    
    for ii, (X, y) in enumerate(data_loader):

        batch_size = X.shape[0]
        seq_len = X.shape[1]
        position = ii*batch_size

        init_hc = None
         
        preds = torch.rand(batch_size)
         
        for idx, thisX in enumerate(X):
             
            if idx == 0:  
   
                position = df.index.get_loc(df_train.index[position])
                prev_x = init_prevX(df, seq_len, position, settings)

            else:

                prev_x = torch.cat((prev_x[1:].flatten(), shoreline_in.detach()), axis = 0)
                prev_x = torch.reshape(prev_x, (seq_len,1))
                prev_x = standardize(prev_x, scalers, 'sl')

            thisX = torch.cat((thisX,prev_x),axis = 1)
            thisX = thisX.unsqueeze(dim=0)
            thispred, init_hc = model(thisX, init_hc)
            
            thispred = destandardize(thispred, scalers, 'dx')
            prev_x = destandardize(prev_x, scalers, 'sl')
            preds[idx] = thispred

            shoreline_in = torch.add(prev_x.flatten()[-1], thispred)
            

        y = destandardize(y, scalers, 'dx') 
        loss = loss_function.forward(preds, y)
         
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
     
    avg_loss = total_loss / num_batches
    return avg_loss 

#####################################################
#####################################################

def init_prevX(df, seq_len, position, settings):

    if seq_len > 1:
        
        #prev_x = torch.tensor(df[settings['shoreline']][position-seq_len:position].values).float().reshape(seq_len,1)

        last_sl_position = df[settings['shoreline']][:position].dropna().values[-seq_len:]
        prev_x = torch.tensor(last_sl_position).float().reshape(seq_len,1)

    else:
        last_valid_idx = df[settings['shoreline']][:position].last_valid_index()
        last_sl_position = df[settings['shoreline']][last_valid_idx]
        prev_x = torch.tensor(last_sl_position).float().reshape(seq_len,1)

    return prev_x
import numpy as np
import pandas as pd
import torch
import os
import math 

from functions.predict import predict_absX

#####################################################
#####################################################

def train_absX(df_train, data_loader, model, loss_function, prev_x, scalers, collect, optimizer):
    '''
    train model w/ loss on absolute shoreline position 
    '''

    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    c_all = []
    position = 0
    for ii, (X, y) in enumerate(data_loader):
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        position = ii*batch_size
        
        if ii <= math.ceil(seq_len/batch_size)-1:
            continue 
         
        preds = torch.rand(batch_size)
         
        for idx, thisX in enumerate(X):
             
            if idx == 0:    
                prev_x = get_history(df_train, seq_len, position)
 
            else:  
                prev_x = torch.cat((prev_x[1:].flatten(), shoreline_in.detach()), axis = 0)
                prev_x = torch.reshape(prev_x, (seq_len,1))
                prev_x = standardize(prev_x, scalers, 'sl')

            thisX = torch.cat((thisX,prev_x),axis = 1)
            thisX = thisX.unsqueeze(dim=0)
            thispred, c_n = model(thisX)

            if collect:
                c_all.append(c_n.flatten().detach().numpy())
            
            thispred = destandardize(thispred, scalers, 'dx')
            prev_x = destandardize(prev_x, scalers, 'sl')
            preds[idx] = thispred

            shoreline_in = torch.add(prev_x.flatten()[-1], thispred)
          
        y = destandardize(y, scalers, 'dx') 
        # loss = loss_function(torch.cumsum(preds, dim =0), torch.cumsum(y, dim =0))
        loss = loss_function(preds, y)
         
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
     
    avg_loss = total_loss / num_batches
    return avg_loss, np.asarray(c_all)

#####################################################
#####################################################

def test_absX(df_train, data_loader, model, loss_function, prev_x, scalers, collect):
    '''
    test model w/ loss on absolute shoreline position 
    '''  
    os.system('cls')
    num_batches = len(data_loader)
    total_loss = 0
    c_all = []

    model.eval()
    with torch.no_grad():

        for ii, (X, y) in enumerate(data_loader):
            batch_size = X.shape[0]
            seq_len = X.shape[1]

            preds = torch.rand(batch_size)
            
            for idx, thisX in enumerate(X):
                
                if ii == 0 and idx == 0:
                    prev_x = get_history_test(df_train, seq_len)
                else:   
                    prev_x = torch.cat((prev_x[1:].flatten(), shoreline_in.detach()), axis = 0)
                    prev_x = torch.reshape(prev_x, (seq_len,1))
                    prev_x = (prev_x - scalers['sl_m'])/scalers['sl_std']
                
                thisX = torch.cat((thisX,prev_x),axis = 1)
                thisX = thisX.unsqueeze(dim=0)
                thispred, c_n = model(thisX)

                if collect:
                    c_all.append(c_n.flatten().detach().numpy())
                
                thispred = destandardize(thispred, scalers, 'dx')
                prev_x = destandardize(prev_x, scalers, 'sl')
                preds[idx] = thispred

                shoreline_in = torch.add(prev_x.flatten()[-1], thispred)

            y = destandardize(y, scalers, 'dx')
            # loss = loss_function(torch.cumsum(preds, dim =0), torch.cumsum(y, dim =0))
            loss = loss_function(preds, y)
    
            total_loss = total_loss + loss.item()
      
    avg_loss = total_loss / num_batches
    return avg_loss,  np.asarray(c_all)

#####################################################
#####################################################

def get_history(df, seq_len, position): 
    prev_x = torch.tensor(df['SL_x'][position-seq_len:position].values).float().reshape(seq_len,1)
    return prev_x

#####################################################

def get_history_test(df, seq_len):
    prev_x = torch.tensor(df['SL_x'][-seq_len:].values).float().reshape(seq_len,1)
    return prev_x

#####################################################

def destandardize(arr, scalers, target):
    _arr_ = (arr*scalers[f'{target}_std'])+scalers[f'{target}_m']
    return _arr_

#####################################################

def standardize(arr, scalers, target):
    _arr_ = (arr - scalers[f'{target}_m'])/scalers[f'{target}_std']
    return _arr_

#####################################################
#####################################################

def train_probe(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:

        # X = torch.cat((X),axis = 2)
        output = model(X)
        output = model(X)
        
        output = torch.flatten(output)
        loss = loss_function(output, y)
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches

    # print(f"Train loss: {avg_loss}")
    return avg_loss

#####################################################
#####################################################

def test_probe(data_loader, model, loss_function):
    os.system('cls')
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            output = torch.flatten(output)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    # lossTest = np.append (lossTest, avg_loss)

    # print(f"Test loss: {avg_loss}")
    return avg_loss
  
    
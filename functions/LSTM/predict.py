# Predict functions...

import torch
from functions.misc import *

#####################################################
#####################################################

def work(df_train, data_loader, model, scalers, start_date, settings):

    output = torch.tensor([])
    model.eval()
    cell_state_vector = []
    init_hc = None

    with torch.no_grad():
         
        for ii, (X, y) in enumerate(data_loader):
            
            batch_size = X.shape[0]
            seq_len = X.shape[1]
            
            preds = torch.rand(batch_size) 
            true_preds = torch.rand(batch_size)

            for idx, thisX in enumerate(X):

                if ii == 0 and idx == 0:
                    prev_x = init_prevX(df_train, seq_len, start_date, settings)
                else:
                    prev_x = torch.cat((prev_x[1:].flatten(), shoreline_in.detach()), axis = 0)
                    prev_x = torch.reshape(prev_x, (seq_len,1))
                    prev_x = standardize(prev_x, scalers, 'sl')


                thisX = torch.cat((thisX,prev_x),axis = 1)
                thisX = thisX.unsqueeze(dim=0)

                thispred, init_hc = model(thisX, init_hc) # predict
                c_n = init_hc[1]
                cell_state_vector.append(c_n.flatten().detach().numpy()) # collect value of the cell states
                
                thispred = destandardize(thispred, scalers, 'dx')
                prev_x = destandardize(prev_x, scalers, 'sl')
                preds[idx] = thispred

                true_preds[idx] = torch.add(prev_x.flatten()[-1], thispred)
                # if idx < len(X)-1:
                shoreline_in = torch.add(prev_x.flatten()[-1], thispred)
                

            output = torch.cat((output,true_preds.flatten()[-batch_size:]), 0)

    return output, cell_state_vector

#####################################################
#####################################################

def init_prevX(df, seq_len, start_date, settings):

    if seq_len > 1:

        #prev_x = torch.tensor(df[settings['shoreline']][:start_date][-seq_len:].values).float().reshape(seq_len,1)

        last_sl_position = df[settings['shoreline']][:start_date].dropna().values[-seq_len:]
        prev_x = torch.tensor(last_sl_position).float().reshape(seq_len,1)

    else: 
        last_valid_idx = df[settings['shoreline']][:start_date].last_valid_index()
        last_sl_position = df[settings['shoreline']][last_valid_idx]
        prev_x = torch.tensor(last_sl_position).float().reshape(seq_len,1)

    return prev_x
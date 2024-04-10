 # model defenition

import torch
import torch.nn as nn
import numpy as np

#####################################################
#####################################################

class LSTM(nn.Module):

    '''
    Stacked LSTM model, dictated by input parameters 
    '''

    def __init__(self, num_sensors, hidden_units, num_layers, dropout=0.4):
        super(LSTM, self).__init__()
        self.num_sensors = num_sensors+1  # this is the number of features + reinput shoreline
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        # self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=num_sensors+1,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=hidden_units, out_features=1)
        
    def forward(self, x, init_hc=None):

        device = next(self.parameters()).device
        
        h0 = torch.zeros(self.num_layers, 1, self.hidden_units, device=device)  # Initialize h0
        c0 = torch.zeros(self.num_layers, 1, self.hidden_units, device=device)  # Initialize c0 
     
        if init_hc is None:
            _, (hn, cn) = self.lstm(x, (h0, c0))
        else:
            hn, cn = init_hc 
            _, (hn, cn) = self.lstm(x, (hn, cn))

        hn = self.dropout(hn[-1])
        out = self.linear(hn).flatten()

        return out, (torch.unsqueeze(hn, 0), cn)

#####################################################
#####################################################

class CustomMSE():
    def  __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, output, target):

        if torch.isnan(target).all():
            return torch.tensor([0.0]).requires_grad_()

        n = target.shape[0]
        cum_y = 0
        MSE = 0
 
        output_ = torch.cumsum(output, axis=0)

        for ii, y in enumerate(target):

            if torch.isnan(y):
                continue
            else:
                cum_y = torch.add(y,cum_y)
                target[ii] = cum_y

                err = torch.square(target[ii]-output_[ii])
                MSE = MSE + err

        MSE = (MSE/n)
        
        return MSE
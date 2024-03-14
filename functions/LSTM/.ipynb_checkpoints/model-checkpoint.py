# model defenition

from unicodedata import bidirectional
import torch
import torch.nn as nn

#####################################################
#####################################################

class LinearProbe(nn.Module):
    def __init__(self, num_sensors, dropout=0.2):
        super(LinearProbe, self).__init__()

        self.num_sensors = num_sensors  # this is the number of features
        self.dropout = dropout

        self.l1 = nn.Linear(num_sensors,111)
        self.l2 = nn.Linear(111,1)
            
    def forward(self,x):
        out = self.l1(x)
        out = self.l2(out)
        return out

class nonLinearProbe(nn.Module):
    def __init__(self, num_sensors, hidden_units_1, hidden_units_2, dropout=0.2, ):
        super(nonLinearProbe, self).__init__()

        self.num_sensors = num_sensors  # this is the number of features
        self.dropout = dropout
        
        self.l1 = nn.Linear(num_sensors,hidden_units_1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_units_1,hidden_units_2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_units_2,1)
            
    def forward(self,x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out

class LSTM(nn.Module):

    '''
    Stacked LSTM model, dictated by input parameters 
    '''

    def __init__(self, num_sensors, hidden_units, num_layers, dropout=0.4):
        super(LSTM, self).__init__()
        self.num_sensors = num_sensors+1  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=num_sensors+1,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=hidden_units, out_features=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        # batch_size = 1
    
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        _, (hn, c_n) = self.lstm(x, (h0, c0)) 
        hn = self.dropout(hn[0])
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out, c_n

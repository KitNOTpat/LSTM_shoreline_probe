# data select

import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import StandardScaler

class data_select():

    def __init__(self, settings):
        self.settings = settings
        self.target = settings['target']

        if settings['case'] != 'synthetic':
            self.df = pd.read_csv(os.path.join(os.getcwd(),'data','input_params', settings['case']+'.csv'), index_col = 'date', parse_dates=True, dayfirst=True)
            self.features = self.df[settings['dynamic_inputs']].columns
        
        if self.target in self.features:
            raise ValueError(f'Target variable {self.target} present within input feature list.')

    def train_test_split(self):

        '''
        Split the data into the train and test. we cut the first `sequence length` of datapoints to allow initiation.
        we also make sure the size of the dataset is divisible by the size of each bit to make sure each bit is the
        same length.
        '''
        sl_var_name = self.settings['shoreline']

        first_val_idx = self.df[sl_var_name].dropna()[self.settings['sequence_length']:].index[0]
        mask = self.df[first_val_idx:].copy()

        split_index = int(0.5 * len(mask))
        self.train = mask[:split_index].copy()
        self.test = mask[split_index:].copy()

        if self.settings['case'] == 'synthetic':
            self.obs = self.train['SL_0.0'].values

    ##############################
    ##############################

    def standardize(self):
        
        dx_m  = self.train[self.settings['target']].mean()
        dx_std = self.train[self.settings['target']].std()
        sl_m = self.train[self.settings['shoreline']].mean()
        sl_std = self.train[self.settings['shoreline']].std()

        self.scalers = {
                'dx_m':dx_m, 
                'dx_std':dx_std, 
                'sl_m':sl_m, 
                'sl_std':sl_std}
         
        for col in self.df.columns:
            
            if col == 'dates' or col == 'date' or col == 'omega_filt' or col == 'err':
                continue
            
            temp_mean = self.train[col].mean()
            temp_std = self.train[col].std()
                    
            self.train[col] = (self.train[col] - temp_mean) / temp_std
            self.test[col] = (self.test[col] - temp_mean) / temp_std
            self.df[col] = (self.df[col] - temp_mean) / temp_std

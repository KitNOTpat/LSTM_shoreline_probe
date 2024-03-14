# data select

import pandas as pd
import os 
import matplotlib.pyplot as plt

class data_select():
    def __init__(self, settings):
        self.settings = settings
        self.target = settings['target']
        self.train_start = pd.to_datetime(self.settings['train_start_date'])
        self.train_end = pd.to_datetime(self.settings['train_end_date'])

        if settings['case'] == 'narra2019':
            self.df = pd.read_csv(os.path.join(os.getcwd(),'data','input_params','2019_narra_force_params_3D.csv'), index_col = 'date', parse_dates=True)
            self.features = self.df[settings['dynamic_inputs']].columns

        elif settings['case'] == 'narra2014':
            self.df = pd.read_csv(os.path.join(os.getcwd(),'data','input_params','narra_force_params_3D.csv'), index_col = 'date', parse_dates=True)
            self.features = self.df[settings['dynamic_inputs']].columns

        elif settings['case'] == 'synthetic':
            self.df = pd.read_csv(os.path.join(os.getcwd(),'data','input_params','syn_R0_phi25.csv'))
            # self.df['date'] = pd.to_datetime(self.df.date, format="%Y-%m-%d")
            self.df['date'] = pd.to_datetime(self.df.date)
            self.df.set_index(['date'], drop =True, inplace = True)
            self.features = self.df[settings['dynamic_inputs']].columns
        
        if self.target in self.features:
            raise ValueError(f'Target variable {self.target} present within input feature list.')

    def train_test_split(self):

        self.train = self.df[self.train_start:self.train_end].copy()
        # self.test = self.df.loc[self.df.index.difference(self.train.index)].copy()

        # print("Train set proportion:", len(self.train) / len(self.df))
        # print("Train start date:", self.settings['train_start_date'])
        # print('Train Shape:',self.train.shape)
        # print('Test Shape:', self.test.shape)

    def standardize(self):

        target_mean = self.train[self.target].mean()
        target_stdev = self.train[self.target].std()

        for col in self.train.columns:
            if col == 'dates' or col == 'date':
                continue
            # if col == 'dt':
            #     continue
            #     df_train[col]= df_train[col] / dt_max
            else: 
                mean = self.train[col].mean()
                stdev = self.train[col].std()
                
                if col == 'dx':
                    dx_m, dx_std = mean, stdev
                if col == 'SL_x':
                    sl_m, sl_std = mean, stdev

                self.train[col] = (self.train[col] - mean) / stdev
                self.test_set_A[col] = (self.test_set_A[col].values - mean) / stdev
                self.test_set_B[col] = (self.test_set_B[col].values - mean) / stdev
                self.df[col] = (self.df[col].values - mean) / stdev
                
        self.scalers = {
                'dx_m':dx_m, 
                'dx_std':dx_std, 
                'sl_m':sl_m, 
                'sl_std':sl_std}
    
    def split_test_sets(self):
        
        self.test_set_A = self.df[:self.train_start].copy()
        self.test_set_A = self.test_set_A[self.settings['sequence_length']:].copy()
        self.test_set_B = self.df[self.train_end:].copy()

        # self.test_set_A.SL_x.plot(figsize = (30,5), c = 'blue')
        # self.test_set_B.SL_x.plot(c = 'blue', label = 'training set')
        # self.train.SL_x.plot(c = 'darkorange', label = 'test set')

        return self.test_set_A, self.test_set_B

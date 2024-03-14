#plotting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class graphs():
    def __init__(self, df, df_out, df_train, target):
        self.df = df
        self.out = df_out
        self.trainlen = len(df_train)
        self.target = target

    def plot_dx(self):

        plt.rc('font',family='Times New Roman')
        plt.figure(figsize=(30, 7))

        # self.out.at[self.trainlen,'Model forecast'] = self.df[self.trainlen:]['SL_x'].values[0]
        # br 
        plt.plot(self.out[self.target], color = 'k',alpha = 0.8, label = 'dx target')
        plt.plot(self.out.index[:self.trainlen], self.out['Model forecast'][:self.trainlen], color='royalblue',linestyle='dashed', label = 'dx train')
        plt.plot(self.out.index[self.trainlen:], self.out['Model forecast'][self.trainlen:], color='r',linestyle='dashed',label = 'dx test')

        plt.ylabel('Shoreline Position (m)', fontsize = 'xx-large')
        plt.ylim([self.out[self.target].min()-1, self.out[self.target].max()+1])
        plt.grid(linestyle = '--', linewidth = 1, axis='both')
        plt.legend( fontsize='xx-large', edgecolor = 'k')

        # plt.axvline(self.trainlen, linestyle='dashed', c = 'k', alpha = 1) # vertical

    def dx_hist(self):
        fig,axs = plt.subplots(1,2, figsize = (17,5))

        bins = np.linspace(-10, 10, 100)

        train_pred = [self.out[:self.trainlen:]['Model forecast'].values]
        train_target = [self.out[:self.trainlen]['dx'].values]
        test_pred = [self.out[self.trainlen:]['Model forecast'].values]
        test_target = [self.out[self.trainlen:]['dx'].values]

        bins = np.linspace(-10, 10, 100)

        axs[0].hist(train_target, bins, density = False, alpha=1, label='measured', color = 'whitesmoke', edgecolor='k')
        axs[1].hist(test_target, bins, density = False, alpha=1, label='measured', color = 'whitesmoke', edgecolor='k')

        axs[0].hist(train_pred, bins, density = False, alpha=1, label='predicted', color = 'dimgrey', edgecolor='k')
        axs[1].hist(test_pred, bins, density = False, alpha=1, label='predicted', color = 'dimgrey', edgecolor='k')
        
        axs[0].set_title('Train', fontsize =15)
        axs[1].set_title('Test', fontsize =15)

        axs[0].legend( fontsize='x-large', edgecolor = 'k')
        axs[1].legend( fontsize='x-large', edgecolor = 'k')
        fig.suptitle('frequency of dx',fontweight ="bold")

    def plot_sl(self):

        train_df = self.out[:self.trainlen].copy()
        train_df.reset_index(inplace=True)
        train_df.at[0,'Model forecast'] = self.df['SL_x'].values[0]
         
        plt.figure(figsize=(30, 7))
        plt.plot(pd.to_datetime(self.df['date']),self.df['SL_x'], color = 'k',alpha = 0.8, label = 'Shoreline')
        plt.plot(pd.to_datetime(self.df[:self.trainlen]['date']),train_df['Model forecast'].cumsum(), color='royalblue',linestyle='dashed', label = 'train')

        test_df = self.out[self.trainlen:].copy()
        test_df.reset_index(inplace=True)
        test_df.at[0,'Model forecast'] = self.df[self.trainlen:]['SL_x'].values[0]

        plt.plot(pd.to_datetime(self.df[self.trainlen:]['date']),test_df['Model forecast'].cumsum(), color='r',linestyle='solid', label = 'test')

        plt.ylabel('Shoreline Position (m)', fontsize = 'xx-large')
        plt.grid(linestyle = '--', linewidth = 1, axis='both')
        plt.legend( fontsize='xx-large', edgecolor = 'k', loc = 2)
        # plt.savefig('results/NARX_LSTM_narra_3D.png')

        return train_df, test_df

    def plot_test(self):
        test_df = self.out[self.trainlen:].copy()
        test_df.reset_index(inplace=True)
        test_df.at[0,'Model forecast'] = self.df[self.trainlen:]['SL_x'].values[0]

        plt.figure(figsize=(30, 7))
        plt.plot(test_df.index , self.df[self.trainlen:]['SL_x'], color = 'k',alpha = 0.8, label = 'Shoreline')
        plt.plot(test_df['Model forecast'], color='r',linestyle='dashed', label = 'Forecast')
        plt.ylabel('Shoreline Position (m)', fontsize = 'xx-large')
        plt.grid(linestyle = '--', linewidth = 1, axis='both')
        plt.legend( fontsize='xx-large', edgecolor = 'k')
        # plt.savefig('results/NARX_LSTM_narra_3D.png')


def learning_curve(lc_train, lc_test):

    plt.figure(figsize=(7, 5))
    plt.rc('font',family='Times New Roman')
    plt.grid(linestyle = '--', linewidth = 1, axis='both')
    plt.plot(lc_train, label = 'Training Loss', color = 'k')
    plt.plot(lc_test, label = 'Test Loss', color = 'k', linestyle = 'dashed')
    plt.ylabel('Loss', fontsize = 'large')
    plt.xlabel('Epoch', fontsize = 'large')
    plt.legend( fontsize='large', edgecolor = 'k')



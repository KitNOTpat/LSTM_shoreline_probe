# LSTM weather forecasting practice using 

# we need to put the input data in input matrix, each row is going to be the input and
# they have a corresponding label

# X = [[[1],[2],[3],[4],[5]]] Y = [6]
# [[2],[3],[4],[5],[6]]] [7]
# we could have a (extra[]) to account for multivariate data, maybe [temp, pressure]

import os
import pandas as pd
import numpy as np
# import seaborn as sns
# import tensorflow as tf
import matplotlib.pyplot as plt 

from operator import attrgetter

#####################################################
#####################################################

def to_dxdt(temp, transect, month_delta):

    temp['dates'] = pd.to_datetime(temp['dates'], format='%Y-%m-%d %H:%M:%S%z')

    if month_delta:
        temp = temp.groupby(pd.PeriodIndex(temp['dates'], freq="M"))[transect].mean()
        temp = pd.DataFrame(temp, index = None)
        temp.reset_index(inplace=True)
        temp['dt'] = (temp['dates'] - temp['dates'][0]).apply(attrgetter('n'))
    else:
        temp['dt'] = (temp['dates'] - temp['dates'][0])
    temp['dx'] = temp[transect].diff()

    # code to merge Hs and dx onto same dataframe
    hs_info = input_Hs()
    temp.set_index('dates', inplace=True)
    hs_info.set_index('dates', inplace=True)
    temp = pd.merge(temp, hs_info, left_index=True, right_index=True)

    temp.reset_index(inplace=True)
    # temp.drop(temp.columns.difference(['dt','dx','Hs']), 1, inplace=True)
    
    temp.drop(temp.columns.difference([transect,'dt','dx','Hs']), 1, inplace=True)
    temp.loc[0, 'dx'] = 0
    temp.dropna(inplace = True)
    temp.rename(columns = {transect:'SL_x'}, inplace = True)
    
    # standardize the input to make sure they are on a similar scale
    # temp['dx'] = (temp['dx']-temp['dx'].mean())/(temp['dx'].max()-temp['dx'].min())
    # temp['dt'] = temp['dt']/temp['dt'].max()

    # visualize data

    # color palette as dictionary
    palette = {"dx":"tab:blue",
            "Hs":"tab:red"}

    plotdf = temp.set_index('dt')
    plotdf['Hs'] = (plotdf['Hs']-plotdf['Hs'].mean())/(plotdf['Hs'].max()-plotdf['Hs'].min())
   
    # sns.set(rc = {'figure.figsize':(20,8)})
    # ax = sns.lineplot(data=plotdf, palette=palette)
    # ax = plotdf.plot(x = 'dt', y = 'dx', figsize = (15,5), xlabel = 'timeDelta', ylabel = 'dx')

    return temp

#####################################################
#####################################################

def input_Hs():

    csv_path = os.path.join(os.getcwd(),'data','real','wave','Inshore_Waves.csv')
    temp = pd.read_csv(csv_path)
    temp = temp.loc[temp['Profile ID']=='PF4']
    temp['dates'] = pd.to_datetime(temp['Date and time (dd/mm/yyyy HH:MM AEST)'])
    temp = temp.groupby(pd.PeriodIndex(temp['dates'], freq="M"))['Significant wave height (m)'].mean()
    temp = pd.DataFrame(temp, index = None)
    temp.reset_index(inplace=True)
    temp.rename(columns = {'Significant wave height (m)':'Hs'}, inplace = True)

    return temp

# merge dataframe w/ data to get corresponding Hs on correct month (Wednesday)

#####################################################
#####################################################


def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X),  np.array(y)

#####################################################
#####################################################

def df_to_X_y_multi(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][1]
        y.append(label)
    return np.array(X),  np.array(y)

#####################################################
#####################################################

def learning_curves(history_dict):
    # learning curve
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)
    # # fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    plt.plot(epochs, loss_values, 'b-', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r-', label='Validation loss')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.show()

#####################################################
#####################################################

def predict(model1, X_train, X_val, X_test, y_train, y_val, y_test):

    train_predictions = model1.predict(X_train).flatten()
    val_predictions = model1.predict(X_val).flatten()
    test_predictions = model1.predict(X_test).flatten()


    train_results = pd.DataFrame(data = {'Train Predictions': train_predictions[:50], 'Actuals': y_train[:50]})
    val_results = pd.DataFrame(data = {'Val Predictions': val_predictions, 'Actuals': y_val})
    test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})

    fig, ax = plt.subplots(1, 3, figsize=(20,5))

    ax[0].plot(train_results['Actuals'][:100],color='red')
    ax[0].plot(train_results['Train Predictions'][:100], color='blue')
    ax[0].set_title("X_Train")
    ax[1].plot(val_results['Actuals'],color='red')
    ax[1].plot(val_results['Val Predictions'], color='blue')
    ax[1].set_title("X_Val")
    ax[2].plot(test_results['Actuals'],color='red')
    ax[2].plot(test_results['Test Predictions'], color='blue')
    ax[2].set_title("X_Test")

#####################################################
#####################################################

def standardize(df_train, df_test):

    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev

    return df_train, df_test


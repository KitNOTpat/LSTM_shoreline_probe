# post-processing to create dataframe that stores results of model

import os
import pandas as pd
import regex as re

from functions.shorefor_utils import *

import plotly.express as px
import plotly.graph_objects as go

#####################################################
#####################################################
def plot_probe(df_out, df, test_start, settings):

    plot_template = dict(
    layout=go.Layout({
        "font_size": 12,
        "xaxis_title_font_size": 12,
        "yaxis_title_font_size": 12})
    )

    fig = px.line(df_out, x=df_out.index, y=[settings.get('target'), 'Model forecast'])
    fig.add_vline(x=test_start, line_width=4, line_dash="dash")

    fig.update_layout(
        template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
    )
    fig.show()

    
    # # df_test = df_out[test_start:].copy()
    # # df_test.reset_index(inplace = True)
    # # df_test.loc[0, 'Model forecast'] = df['SL_x'][test_start]
    # # df_test['SL_forecast'] = df_test['Model forecast'].cumsum()

    # df_out = df_out[:test_start].copy()
    # df_out.loc[0, 'dx'] = df['SL_x'][0]
    # # df_out.loc[test_start, 'dx'] = df['SL_x'][0]
    # df_out.loc[0, 'Model forecast'] = df['SL_x'][0]
    # df_out['SL_forecast'] = df_out['Model forecast'].cumsum()

    # df_out = pd.concat([df_out, df_test])
    # df_out.reset_index(inplace = True)
    # df_out['SL_x'] = df['SL_x']
    # df_out['dt'] = df['dt']

    # plot_template1 = dict(
    # layout=go.Layout({
    #     "font_size": 12,
    #     "xaxis_title_font_size": 12,
    #     "xaxis_title": 'X',
    #     "yaxis_title_font_size": 12})
    # )

    # fig1 = px.line(df_out, x='dt', y=['SL_x','SL_forecast'])
    # fig1.add_vline(x=df.iloc[test_start]['dt'], line_width=4, line_dash="dash")
    # fig1.update_layout(
    #     template=plot_template1, legend=dict(orientation='h', y=1.02, title_text="")
    # )
    # fig1.show()

    return df_out


def plot_result(df_out, df, test_start, settings):

    plot_template = dict(
    layout=go.Layout({
        "font_size": 12,
        "xaxis_title_font_size": 12,
        "yaxis_title_font_size": 12})
    )

    fig = px.line(df_out, x=df_out.index, y=[settings.get('target'), 'Model forecast'])
    fig.add_vline(x=test_start, line_width=4, line_dash="dash")

    fig.update_layout(
        template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
    )
    fig.show()

    
    df_test = df_out[test_start:].copy()
    df_test.reset_index(inplace = True)
    df_test.loc[0, 'Model forecast'] = df['SL_x'][test_start]
    df_test['SL_forecast'] = df_test['Model forecast'].cumsum()

    df_out = df_out[:test_start].copy()
    df_out.loc[0, 'dx'] = df['SL_x'][0]
    # df_out.loc[test_start, 'dx'] = df['SL_x'][0]
    df_out.loc[0, 'Model forecast'] = df['SL_x'][0]
    df_out['SL_forecast'] = df_out['Model forecast'].cumsum()

    df_out = pd.concat([df_out, df_test])
    df_out.reset_index(inplace = True)
    df_out['SL_x'] = df['SL_x']
    df_out['dt'] = df['dt']

    plot_template1 = dict(
    layout=go.Layout({
        "font_size": 12,
        "xaxis_title_font_size": 12,
        "xaxis_title": 'X',
        "yaxis_title_font_size": 12})
    )

    fig1 = px.line(df_out, x='dt', y=['SL_x','SL_forecast'])
    fig1.add_vline(x=df.iloc[test_start]['dt'], line_width=4, line_dash="dash")
    fig1.update_layout(
        template=plot_template1, legend=dict(orientation='h', y=1.02, title_text="")
    )
    fig1.show()

    return df_out


def store_result(model, lossTest, SL_loss, settings, hyper_params, transect):

    record = pd.read_csv(os.path.join(os.getcwd(),'results','run_history.csv'))
    # run_number = 
    label = settings.get('file_name').split('.')[0]
    R, dt = re.findall('[0-9]+', settings.get('file_name'))
     
    result_dict = {
        'label': label if settings.get('synthetic') else transect,
        'synthetic': settings.get('synthetic'),
        'forcing_params':settings.get('forcing') if settings.get('synthetic') else 'N/A',
        'model': model.__class__.__name__,
        'architecture': '{}_{}_{}'.format(
            hyper_params.get('stacked_lstm_no'),
            hyper_params.get('num_hidden_units'),
            hyper_params.get('neuron_dropout')),
        'epochs': hyper_params.get('epochs'),
        'batch_size': hyper_params.get('batch_size'),
        'alpha': hyper_params.get('learning_rate'),
        'noise': R if settings.get('synthetic') else 'N/A',
        'interval':dt if settings.get('synthetic') else 'N/A',
        'data_dropout': settings.get('data_dropout') if settings.get('synthetic') else 'N/A',
        'mse_dx': lossTest,
        'rmse_SL': SL_loss,
        
        # 'runtime'
        }

    record.drop(record.columns.difference(result_dict.keys()), 1, inplace=True)
    record = record.append(result_dict,ignore_index=True)
    return record


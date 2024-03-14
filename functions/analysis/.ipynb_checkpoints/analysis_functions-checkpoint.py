import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import random
from datetime import datetime, timedelta
import torch.nn as nn
import torch

###############################################################################
###############################################################################

def generate_storm_dataset(shl_data,wave_data,storm_thresh=None):
    '''
    Take in the shoreline and wave data and generate a dataset of 
    shoreline movement per storm.
    We are a little bit cheeky here and group consecutive
    shoreline movements together with the hope of getting a more 
    representative storm envelope.
    '''
    # Prepare the diffstorm_data df
    storm_data = shl_data.copy()

    # XX perhaps here we could take the mean of the previous X dates
    storm_data = storm_data.diff()
    storm_data.columns = ['dShl']
    # define dShl as positive for recession
    storm_data.loc[:,'dShl'] = -storm_data['dShl']

    storm_data['postDate'] = storm_data.index
    storm_data.loc[storm_data.index[1:],'preDate'] = storm_data.index[:-1]
    storm_data.drop(storm_data.index[0],inplace=True)
    storm_data.index = ['Storm_{0:04.0f}'.format(_) for _ in range(storm_data.shape[0])]

    # use the > 95% rule if no threshold is specified
    if storm_thresh is None:
        storm_thresh = wave_data['Hsig'].quantile(0.95)

    # split the wave data according to the shoreline data
    waveData = {}
    for thisStorm in storm_data.index:
        waveData[thisStorm] = wave_data.loc[storm_data.loc[thisStorm,'preDate']:storm_data.loc[thisStorm,'postDate']]
        storm_data.loc[thisStorm,'E'] = calculate_wave_energy(waveData[thisStorm],storm_thresh=storm_thresh)

    # constrain storm_data by adding consective shoreline movements
    storm_data['timeDelta'] = (storm_data['postDate'] - storm_data['preDate']).dt.days
    storm_data['zeroCross'] = np.sign(storm_data['dShl']).diff().ne(0).astype(int)
    storm_data['zeroCross'] = storm_data['zeroCross'].cumsum()
    groupedVals = storm_data.groupby(by='zeroCross',as_index=False).sum()
    groupedVals.index = storm_data.index[[np.where(storm_data['zeroCross'] == _)[0][0] for _ in groupedVals['zeroCross']]]
    storm_data['E'] = groupedVals['E']
    storm_data['dShl'] = groupedVals['dShl']
    storm_data['timeDelta'] = groupedVals['timeDelta']
    storm_data.drop_duplicates(subset=['zeroCross'],inplace=True)

    return storm_data, storm_thresh


###############################################################################
###############################################################################

def calculate_wave_energy(waveIn,storm_thresh=3.0):
    '''
    Calculate the wave energy for a storm - see Harley et al. 2009
    '''
    rho = 1025 # kg/m^3
    g = 9.81 # m/s^2
    # this needs to be more robust
    dt = (waveIn.index[1] - waveIn.index[0]).seconds/3600
    stormCriteria = waveIn['Hsig']>storm_thresh #m
    energy = np.sum(waveIn.loc[stormCriteria,'Hsig']**2)*rho*g*dt*(1/16)
    return energy


###############################################################################
###############################################################################

def calculate_storm_energy(hsig_max, storm_duration, storm_thresh=3.0):
    '''
    Using this, the energy will be calculated for a triangular storm event. 
    We will assume the wave height to be defined as a storm is $H_{sig} = 3$ m 
    and the storm will increase in wave height to peak halfway through the event 
    with the specified maximum $H_{sig}$.
    '''
    storm_duration = int(storm_duration)
    dummy_time = pd.to_datetime('2022-01-01 00:00:00')
    dummy_delta = pd.to_timedelta(storm_duration,unit='H')
    times = pd.date_range(dummy_time,dummy_time+dummy_delta,freq='H')
    #simple triangular storm
    uptick = np.floor(storm_duration/2).astype(int)
    waves = np.zeros((storm_duration,))
    waves[:uptick] = np.linspace(storm_thresh,hsig_max,uptick)
    waves[uptick:] = np.linspace(hsig_max,storm_thresh,storm_duration-uptick)

    wave_data = pd.DataFrame(
        {'Hsig':waves},
        index=times[:-1]
    )
    # now use our handy energy calc above
    dummy_energy = calculate_wave_energy(wave_data)
    return dummy_energy

###############################################################################
###############################################################################

def find_pareto_front(data):
    '''
    Find the pareto front of a dataframe - we have a simple 2D 
    case where we know the variables so brute force it the 
    hard coded lazy way (:
    '''
    data['pareto'] = 0
    prevE = 0
    prevShl = -1e9
    data.sort_values(by='E',ascending=True,inplace=True)
    # roll on through and mark the pareto front
    for ind, row in data.iterrows():
        if row['E'] > prevE and row['dShl'] > prevShl:
            data.loc[ind,'pareto'] = 1
            prevE = row['E']
            prevShl = row['dShl']
    # and then give distance to pareto front

    data['paretoDistance'] = 0
    # log transform and scale before we calc dist so its not dominated by E and
    # removes some underlying heteroscedasticity
    scale_data = data.copy()
    scale_data['E'] = np.log(scale_data['E'])
    scale_data['dShl'] = np.log(scale_data['dShl'])
    scale_data.loc[:,'E'] = (scale_data['E'] - scale_data['E'].min())/(scale_data['E'].max() - scale_data['E'].min())
    scale_data.loc[:,'dShl'] = (scale_data['dShl'] - scale_data['dShl'].min())/(scale_data['dShl'].max() - scale_data['dShl'].min())
    paretoPoints = scale_data.loc[data['pareto'] == 1,['E','dShl']].values
    eughPoints = scale_data.loc[data['pareto'] == 0,['E','dShl']].values

    # now get a representative distance from the pareto front and return
    pDists = cdist(paretoPoints,eughPoints,'seuclidean').min(axis=0)
    data.loc[:,'pareto'] = data['pareto']
    data.loc[data['pareto'] == 0,'paretoDistance'] = pDists
    return data

###############################################################################
###############################################################################

def split_test(test, intrvl):

    '''
    split the test set into predefined intervals. 
    '''
    
    if intrvl == 1:
        return [test]

    start_date = test.index[0]
    end_date = test.index[-1]
    diff = (end_date  - start_date ) / intrvl
    
    dates = []
    tests = []

    for ii in np.arange(1,intrvl,1):
        dates.append(start_date+ii*diff)

    last_check = start_date

    for int in np.arange(0,intrvl,1):

        tests.append(test[last_check:last_check+diff])
        last_check = dates[int-1]

    return tests

###############################################################################
###############################################################################

def select_random_test(test,random_dates,days):
    '''
    Selects random dates and 
    '''

    tests = []

    for date in random_dates:
        tests.append(test[date:date+timedelta(days = days)])

    return tests

###############################################################################
###############################################################################


def rmse(input, SL):

    loss = nn.MSELoss(reduction = 'mean')

    target = torch.tensor(SL[input.index.min():input.index.max()])
    input = torch.tensor(input.mean(axis=1))
    Shoreline_loss = np.sqrt(loss(input,target).item())

    return Shoreline_loss

###############################################################################
###############################################################################

def calculate_wave_energy(waveIn,storm_thresh=3.0):
    '''
    Calculate the wave energy for a storm - see Harley et al. 2009
    '''
    rho = 1025 # kg/m^3
    g = 9.81 # m/s^2
    # this needs to be more robust
    dt = 86400*30

    stormCriteria = waveIn>storm_thresh #m

    energy = np.sum(waveIn.loc[stormCriteria]**2)*rho*g*dt*(1/16)
    # return energy
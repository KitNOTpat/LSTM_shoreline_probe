# linear probe

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#####################################################
#####################################################

def linearProbe(ProbeInData, settings):

    ProbeInData.dropna(axis=0, inplace = True)
    columns = ProbeInData.columns

    x = ProbeInData.loc[:, columns != settings['probe_target']].to_numpy()
    y = ProbeInData[settings['probe_target']].to_numpy()

    linear_model = LinearRegression().fit(x, y)

    pred = linear_model.predict(x)

    return r2_score(y, pred), linear_model
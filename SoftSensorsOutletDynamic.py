# -*- coding: utf-8 -*-

"""
@author Abhilash
August 2021
"""

import os
import pandas as pd
import numpy as np
import datetime


from scipy.integrate import odeint
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt


# find the starting point from which removal percentages should be calculated
data = pd.read_excel('DataEstimate.xlsx')

data['Y2_Predict'] = data['Y2'] # add a blanc data with initial estimates


def ModelTPO(y, t, param, coeff): #rate kinetics, equation 4 of the article
    dydt    = (param['FLOW']*3.6*(param['TPI']-y)/coeff[0]) - coeff[4]*coeff[0]*max(0,(coeff[1]+coeff[2]*param['PAXRATIO']+coeff[3]*param['SSO']))*y/(param['FLOW']*3.6)
    return dydt

def obj_fun(x): # objective function which is to be minimized by the model
    TPO_Estim = data['Y2'][0]
    for i in range(1,len(data)):
        TPI             = data['Y1_Predict'][i-1]
        TPO_prev        = data['Y2_Predict'][i-1]
        Q               = max(50,data['X0'][i-1])
        PAX             = max(150,data['X6'][i-1])
        SSO             = max(7,data['X8'][i-1])
                             
        param           = {'TPI':TPI,'PAXRATIO':PAX/Q,'FLOW':Q,'SSO':SSO}
        integrationTime = (datetime.datetime.strptime(data['Timestamp'][i], '%m/%d/%Y')-datetime.datetime.strptime(data['Timestamp'][i-1], '%m/%d/%Y')).days*24
        tStart, tEnd, N_POINTS = 0, integrationTime, 500
        ts              = np.linspace(tStart, tEnd, N_POINTS)
        coeff           = x 
        
        sol1            = odeint(ModelTPO, TPO_prev, ts, args=(param,coeff))
        data['Y2_Predict'][i] = sol1[-1]
            
    
        
    # calculate the mse between lab and online estimated values
    mse = sum([abs(i-j) for i,j in zip(data['Y2'],data['Y2_Predict'])])
    print('The objective for the current iteration is = {}'.format(mse))
    return mse

# run this section if you want to calibrate the model---------------------------------------------------------------
optim_method = {'1':'Nelder-Mead',
                     '2':'Powell',
                     '3':'BFGS',
                     '4':'SLSQP'
             }

method_selction = 6 # choose your minimization algorithm

param_init = 1640, 5.334, 15.408888, -0.292577955, 0.1
#param_init = 1639.66281, -2.47492823e+00, -3.81658030e-01,  1.42874170e-01, 3.96560920e+00

for i in optim_method:
    res = minimize(obj_fun, param_init, method=optim_method[str(method_selction)], options={'disp': True})


fig, axes = plt.subplots(1, 1, figsize=(8, 5))

axes.scatter(data['Y2'], data['Y2_Predict'], color='red', s=20, label='Prediction')
axes.plot(data['Y2'], data['Y2'], '#1fbdbd',
                label='Ideal Prediction')
axes.set(xlabel='True Data',
            ylabel='Predicted Values', title= 'Least-Square: R = ' + str(round(res.fun,3)))
axes.set_title('Total Phosphorus outlet')

axes.grid
axes.legend()
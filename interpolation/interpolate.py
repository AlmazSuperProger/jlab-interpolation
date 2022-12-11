#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import unicode_literals

import cgi
import re
import base64
import sys
import re
import os.path
import csv
import math
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot
from plotly.graph_objs import Scatter
import math

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from numpy import linspace, zeros, array
from scipy.interpolate import griddata

particle_class = "Pi0P"
w_user = "1.5"
q2_user = "1"
cos_user = "empty"
e_beam_user = "2"
eps_user = "hello"
phi_user = "120"
interp_step_user = "0.1"

df = pd.read_csv('final_table.csv', header=None, sep='\t',
                 names=['Channel', 'MID', 'Wmin', 'Wmax', 'Q2min', 'Q2max', 'Cos(theta)', 'Sigma_T', 'dSigma_T',
                        'Sigma_L', 'dSigma_L', 'Sigma_TT', 'dSigma_TT', 'Sigma_LT', 'dSigma_LT', 'eps'])
df = df.assign(w_average=((df['Wmin'] + df['Wmax']) / 2))
df = df.assign(q2_average=((df['Q2min'] + df['Q2max']) / 2))

if particle_class == 'Pin':
    partNum = '1212'
    ParticleSecret = 'PIN'
    ParticleBeauty = 'gvp--->ПЂвЃєn'
    dataframe = df[(df.Channel == 8) | (df.Channel == 14) | (df.Channel == 41) | (df.Channel == 141)].copy()
    dataframes = [dataframe[(dataframe['w_average'] >= 1.1) & (dataframe['w_average'] <= 1.4) &
                            (dataframe['q2_average'] >= 0.2) & (dataframe['q2_average'] <= 0.7)],
                  dataframe[(dataframe['w_average'] >= 1.4) & (dataframe['w_average'] <= 1.6) &
                            (dataframe['q2_average'] >= 0.2) & (dataframe['q2_average'] <= 0.7)],
                  dataframe[(dataframe['w_average'] >= 1.1) & (dataframe['w_average'] <= 1.6) &
                            (dataframe['q2_average'] >= 1.5) & (dataframe['q2_average'] <= 5)],
                  dataframe[(dataframe['w_average'] >= 1.6) & (dataframe['w_average'] <= 2.1) &
                            (dataframe['q2_average'] >= 1.5) & (dataframe['q2_average'] <= 5)]
                  ]
elif particle_class == 'Pi0P':
    PartNum = '1213'
    ParticleSecret = 'PI0P'
    ParticleBeauty = 'gvp--->ПЂвЃ°p'
    dataframe = df[(df.Channel == 9) | (df.Channel == 37) | (df.Channel == 170)].copy()
    dataframes = [dataframe[(dataframe['w_average'] >= 1) & (dataframe['w_average'] <= 1.8) &
                            (dataframe['q2_average'] >= 0.3) & (dataframe['q2_average'] <= 1.9)],
                  dataframe[(dataframe['w_average'] >= 1) & (dataframe['w_average'] <= 4) &
                            (dataframe['q2_average'] >= 2.9) & (dataframe['q2_average'] <= 6.1)]
                  ]

# Interpolation
# method - 0: empty
# method - 1: W and Q2 filled
# method - 2: W and cos filled
# method - 3: Q2 and cos filled


# Calculation unpolarized
# method - 1: E filled
# method - 2: e_beam_filled

# Calculation cross-section
# method - 0: empty
# method - 1: phi filled


values = [w_user, q2_user, cos_user, e_beam_user, eps_user, phi_user, interp_step_user]
our_method = []

interpolation_method = 0
calc_u_method = 0
calc_cross_section_method = 0

for el in values:
    try:
        _ = float(el)
        our_method.append(1)
    except:
        our_method.append(0)

if our_method[0:3] == [1, 1, 1]:  # W, Q2, cos filled
    interpolation_method = 4
    x_axis_values = np.linspace(-180, 180, 100)
    x_axis_name = "phi"
elif our_method[0:3] == [1, 1, 0]:  # W, Q2 filled
    interpolation_method = 1
    x_axis_values = np.linspace(-1, 1, 100)
    columns = []
    for data in dataframes:
        columns.append([data['w_average'].tolist(), data['q2_average'].tolist(), data['Cos(theta)'].tolist(),
                        data['Sigma_T'].tolist(), data['dSigma_T'].tolist(), data['Sigma_L'].tolist(),
                        data['dSigma_L'].tolist(), data['Sigma_TT'].tolist(), data['dSigma_TT'].tolist(),
                        data['Sigma_LT'].tolist(), data['dSigma_LT'].tolist()])
    val_1 = w_user
    val_2 = q2_user
    x_axis_name = "cos(theta)"
elif our_method[0:3] == [1, 0, 1]:  # W, cos filled
    interpolation_method = 2
    x_axis_values = np.linspace(0, 6, 100)
    columns = []
    for data in dataframes:
        columns.append([data['w_average'].tolist(), data['Cos(theta)'].tolist(), data['q2_average'].tolist(),
                        data['Sigma_T'].tolist(), data['dSigma_T'].tolist(), data['Sigma_L'].tolist(),
                        data['dSigma_L'].tolist(), data['Sigma_TT'].tolist(), data['dSigma_TT'].tolist(),
                        data['Sigma_LT'].tolist(), data['dSigma_LT'].tolist()])
    val_1 = w_user
    val_2 = cos_user
    x_axis_name = "q2(gev2)"
elif our_method[0:3] == [0, 1, 1]:  # Q2, cos filled
    interpolation_method = 3
    x_axis_values = np.linspace(0, 6, 100)
    columns = []
    for data in dataframes:
        columns.append([data['q2_average'].tolist(), data['Cos(theta)'].tolist(), data['w_average'].tolist(),
                        data['Sigma_T'].tolist(), data['dSigma_T'].tolist(),
                        data['Sigma_L'].tolist(), data['dSigma_L'].tolist(),
                        data['Sigma_TT'].tolist(), data['dSigma_TT'].tolist(),
                        data['Sigma_LT'].tolist(), data['dSigma_LT'].tolist()])
    val_1 = q2_user
    val_2 = cos_user
    x_axis_name = "w(gev)"

if interpolation_method != 0:
    if our_method[3] == 1:
        calc_u_method = 1
    elif our_method[4] == 1:
        calc_u_method = 2
if (interpolation_method != 0) and (calc_u_method != 0):
    if our_method[5] == 1:
        calc_cross_section_method = 1




res_x_axis_values = []
res_sigma_TT = []
res_sigma_LT = []
res_sigma_T = []
res_sigma_L = []
res_dsigma_TT = []
res_dsigma_LT = []
res_dsigma_T = []
res_dsigma_L = []

def interpolate_in_one_region(clmns, val_1, val_2, x_axis_values):
    sigma_TT = griddata((clmns[0], clmns[1], clmns[2]),
                        clmns[7], (val_1, val_2, x_axis_values), method='linear', rescale=True)
    sigma_LT = griddata((clmns[0], clmns[1], clmns[2]),
                        clmns[9], (val_1, val_2, x_axis_values), method='linear', rescale=True)
    sigma_T = griddata((clmns[0], clmns[1], clmns[2]),
                       clmns[3], (val_1, val_2, x_axis_values), method='linear', rescale=True)
    sigma_L = griddata((clmns[0], clmns[1], clmns[2]),
                       clmns[5], (val_1, val_2, x_axis_values), method='linear', rescale=True)
    dsigma_TT = griddata((clmns[0], clmns[1], clmns[2]),
                         clmns[8], (val_1, val_2, x_axis_values), method='linear', rescale=True)
    dsigma_LT = griddata((clmns[0], clmns[1], clmns[2]),
                         clmns[10], (val_1, val_2, x_axis_values), method='linear', rescale=True)
    dsigma_T = griddata((clmns[0], clmns[1], clmns[2]),
                        clmns[4], (val_1, val_2, x_axis_values), method='linear', rescale=True)
    dsigma_L = griddata((clmns[0], clmns[1], clmns[2]),
                        clmns[6], (val_1, val_2, x_axis_values), method='linear', rescale=True)

    nans = np.isnan(sigma_TT)
    sigma_TT = sigma_TT[nans == False]
    sigma_LT = sigma_LT[nans == False]
    sigma_T = sigma_T[nans == False]
    sigma_L = sigma_L[nans == False]
    dsigma_TT = dsigma_TT[nans == False]
    dsigma_LT = dsigma_LT[nans == False]
    dsigma_T = dsigma_T[nans == False]
    dsigma_L = dsigma_L[nans == False]
    x_axis_values = x_axis_values[nans == False]

    return (x_axis_values, sigma_TT, sigma_LT, sigma_T, sigma_L, dsigma_TT, dsigma_LT, dsigma_T, dsigma_L)
for clmn in columns:
    tmp_res = interpolate_in_one_region(clmn, val_1, val_2, x_axis_values)
    res_x_axis_values = res_x_axis_values + list(tmp_res[0])
    res_sigma_TT = res_sigma_TT + list(tmp_res[1])
    res_sigma_LT = res_sigma_LT + list(tmp_res[2])
    res_sigma_T = res_sigma_T + list(tmp_res[3])
    res_sigma_L = res_sigma_L + list(tmp_res[4])
    res_dsigma_TT = res_dsigma_TT + list(tmp_res[5])
    res_dsigma_LT = res_dsigma_LT + list(tmp_res[6])
    res_dsigma_T = res_dsigma_T + list(tmp_res[7])
    res_dsigma_L = res_dsigma_L + list(tmp_res[8])




print(len(res_x_axis_values))
print(len(res_sigma_TT))
print(res_x_axis_values)
print(res_sigma_TT)

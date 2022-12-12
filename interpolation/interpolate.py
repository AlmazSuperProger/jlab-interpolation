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
from scipy.interpolate import griddata




gettext = cgi.FieldStorage()
particle_class = gettext.getfirst("particle", "empty")
w_user = gettext.getfirst("w", "empty")
q2_user = gettext.getfirst("q2", "empty")
cos_user = gettext.getfirst("cos", "empty")
e_beam_user = gettext.getfirst("eBeam", "empty")
eps_user = gettext.getfirst("eps", "empty")
phi_user = gettext.getfirst("phi", "empty")
interp_step_user = gettext.getfirst("grid_step_user", "empty")
x_axis_min = gettext.getfirst("this_min_value", "empty")
x_axis_max = gettext.getfirst("this_max_value", "empty")
x_axis_label = "empty"




particle_class = "Pin"
w_user = "1.3"
q2_user = "0.5"
cos_user = "empty"
e_beam_user = "empty"
eps_user = "0.92"
phi_user = "5.75"
interp_step_user = float("0.1")
x_axis_min = "empt"
x_axis_max = "val"
x_axis_label = "empty"
mp = 0.93827

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
# method - 0: E filled
# method - 1: e_beam_filled

# Calculation cross-section
# method - 0: empty
# method - 1: phi filled


values = [w_user, q2_user, cos_user, e_beam_user, eps_user, phi_user, x_axis_min, x_axis_max, interp_step_user,
          x_axis_label]
our_method = []

interpolation_method = -1
calc_u_method = -1
calc_cross_section_method = -1

for el_idx, el in enumerate(values):
    try:
        values[el_idx] = float(el)
        our_method.append(1)
    except:
        our_method.append(0)

columns = []

if our_method[0:3] == [1, 1, 1]:  # W, Q2, cos filled
    interpolation_method = 0
    if our_method[6:8] == [1, 1]:
        values[5] = np.arange(values[6], values[7], 5).tolist()
    else:
        values[5] = np.arange(-180, 180, 5).tolist()
    for data in dataframes:
        columns.append([data['w_average'].tolist(), data['q2_average'].tolist(), data['Cos(theta)'].tolist(),
                        data['Sigma_T'].tolist(), data['dSigma_T'].tolist(), data['Sigma_L'].tolist(),
                        data['dSigma_L'].tolist(), data['Sigma_TT'].tolist(), data['dSigma_TT'].tolist(),
                        data['Sigma_LT'].tolist(), data['dSigma_LT'].tolist()])
    values[-1] = "phi"
    # values[2]=[values[2]]
elif our_method[0:3] == [1, 1, 0]:  # W, Q2 filled
    interpolation_method = 1
    # values[0:2] = [values[0] , values[1]]
    values[-1] = "cos(theta)"
    if our_method[6:8] == [1, 1]:
        values[2] = np.arange(values[6], values[7] + 0.01, values[8])
    else:
        values[2] = np.arange(-1, 1, values[8])
    columns = []
    for data in dataframes:
        columns.append([data['w_average'].tolist(), data['q2_average'].tolist(), data['Cos(theta)'].tolist(),
                        data['Sigma_T'].tolist(), data['dSigma_T'].tolist(), data['Sigma_L'].tolist(),
                        data['dSigma_L'].tolist(), data['Sigma_TT'].tolist(), data['dSigma_TT'].tolist(),
                        data['Sigma_LT'].tolist(), data['dSigma_LT'].tolist()])
elif our_method[0:3] == [1, 0, 1]:  # W, cos filled
    interpolation_method = 2
    values[0:2] = [values[0], values[2]]
    values[-1] = "q2(gev2)"
    if our_method[6:8] == [1, 1]:
        values[2] = np.arange(values[6], values[7] + 0.01, values[8])
    else:
        values[2] = np.arange(0, 6, values[8])
    for data in dataframes:
        columns.append([data['w_average'].tolist(), data['Cos(theta)'].tolist(), data['q2_average'].tolist(),
                        data['Sigma_T'].tolist(), data['dSigma_T'].tolist(), data['Sigma_L'].tolist(),
                        data['dSigma_L'].tolist(), data['Sigma_TT'].tolist(), data['dSigma_TT'].tolist(),
                        data['Sigma_LT'].tolist(), data['dSigma_LT'].tolist()])
elif our_method[0:3] == [0, 1, 1]:  # Q2, cos filled
    interpolation_method = 3
    values[0:2] = [values[1], values[2]]
    values[-1] = "w(gev)"
    if our_method[6:8] == [1, 1]:
        values[2] = np.arange(values[6], values[7] + 0.01, values[8])
    else:
        values[2] = np.arange(0, 6, values[8])

    for data in dataframes:
        columns.append([data['q2_average'].tolist(), data['Cos(theta)'].tolist(), data['w_average'].tolist(),
                        data['Sigma_T'].tolist(), data['dSigma_T'].tolist(),
                        data['Sigma_L'].tolist(), data['dSigma_L'].tolist(),
                        data['Sigma_TT'].tolist(), data['dSigma_TT'].tolist(),
                        data['Sigma_LT'].tolist(), data['dSigma_LT'].tolist()])

if interpolation_method != -1:
    if our_method[3] == 1:
        calc_u_method = 0
    elif our_method[4] == 1:
        calc_u_method = 1
if (interpolation_method != -1) and (calc_u_method != -1):
    if our_method[5] == 1:
        calc_cross_section_method = 1

res_x_axis_values = []
res_sigma_TT, res_sigma_LT, res_sigma_T, res_sigma_L = [], [], [], []
res_dsigma_TT, res_dsigma_LT, res_dsigma_T, res_dsigma_L, = [], [], [], []


def interpolate_in_one_region(clmns, val_1, val_2, x_axis_values):
    base_data = (clmns[0], clmns[1], clmns[2])
    find_points = (val_1, val_2, x_axis_values)

    sigma_tt = griddata(base_data, clmns[7], find_points, method='linear', rescale=True)
    sigma_lt = griddata(base_data, clmns[9], find_points, method='linear', rescale=True)
    sigma_t = griddata(base_data, clmns[3], find_points, method='linear', rescale=True)
    sigma_l = griddata(base_data, clmns[5], find_points, method='linear', rescale=True)
    dsigma_tt = griddata(base_data, clmns[8], find_points, method='linear', rescale=True)
    dsigma_lt = griddata(base_data, clmns[10], find_points, method='linear', rescale=True)
    dsigma_t = griddata(base_data, clmns[4], find_points, method='linear', rescale=True)
    dsigma_l = griddata(base_data, clmns[6], find_points, method='linear', rescale=True)

    if type(x_axis_values) == float:
        x_axis_values = np.array([x_axis_values])
        sigma_tt, sigma_lt, sigma_t, sigma_l = np.array([sigma_tt]), np.array([sigma_lt]), \
                                               np.array([sigma_t]), np.array([sigma_l])
        dsigma_tt, dsigma_lt, dsigma_t, dsigma_l = np.array([dsigma_tt]), np.array([dsigma_lt]), \
                                                   np.array([dsigma_t]), np.array([dsigma_l])
    not_nans = ~np.isnan(sigma_tt.copy())
    return (x_axis_values[not_nans].tolist(),
            sigma_tt[not_nans].tolist(), sigma_lt[not_nans].tolist(),
            sigma_t[not_nans].tolist(), sigma_l[not_nans].tolist(),
            dsigma_tt[not_nans].tolist(), dsigma_lt[not_nans].tolist(),
            dsigma_t[not_nans].tolist(), dsigma_l[not_nans].tolist())


for clmn in columns:
    tmp_res = interpolate_in_one_region(clmn, values[0], values[1], values[2])
    res_x_axis_values = res_x_axis_values + tmp_res[0]
    res_sigma_TT = res_sigma_TT + tmp_res[1]
    res_sigma_LT = res_sigma_LT + tmp_res[2]
    res_sigma_T = res_sigma_T + tmp_res[3]
    res_sigma_L = res_sigma_L + tmp_res[4]
    res_dsigma_TT = res_dsigma_TT + tmp_res[5]
    res_dsigma_LT = res_dsigma_LT + tmp_res[6]
    res_dsigma_T = res_dsigma_T + tmp_res[7]
    res_dsigma_L = res_dsigma_L + tmp_res[8]

res_df = pd.DataFrame({'x_axis_values': res_x_axis_values,
                       'sigma_TT': res_sigma_TT,
                       'sigma_LT': res_sigma_LT,
                       'sigma_T': res_sigma_T,
                       'sigma_L': res_sigma_L,
                       'dsigma_TT': res_dsigma_TT,
                       'dsigma_LT': res_dsigma_LT,
                       'dsigma_T': res_dsigma_T,
                       'dsigma_L': res_dsigma_L})

if interpolation_method == 0:
    if len(res_df) > 0:
        len_val_5 = len(values[5])
        print(values[5])
        print(len(values[5]))
        res_df = pd.DataFrame({'x_axis_values': values[5],
                               'sigma_TT': res_sigma_TT * len_val_5,
                               'sigma_LT': res_sigma_LT * len_val_5,
                               'sigma_T': res_sigma_T * len_val_5,
                               'sigma_L': res_sigma_L * len_val_5,
                               'dsigma_TT': res_dsigma_TT * len_val_5,
                               'dsigma_LT': res_dsigma_LT * len_val_5,
                               'dsigma_T': res_dsigma_T * len_val_5,
                               'dsigma_L': res_dsigma_L * len_val_5})

        res_df['w_values'], res_df['q2_values'], res_df['cos_values'] = [values[0]] * len_val_5, \
                                                                        [values[1]] * len_val_5, \
                                                                        [values[2]] * len_val_5
elif interpolation_method == 1:
    res_df['w_values'], res_df['q2_values'], res_df['cos_values'] = [values[0]] * len(res_df), \
                                                                    [values[1]] * len(res_df), \
                                                                    res_df['x_axis_values']
elif interpolation_method == 2:
    res_df['w_values'], res_df['q2_values'], res_df['cos_values'] = [values[0]] * len(res_df), \
                                                                    res_df['x_axis_values'], \
                                                                    [values[1]] * len(res_df)
elif interpolation_method == 3:
    res_df['w_values'], res_df['q2_values'], res_df['cos_values'] = res_df['x_axis_values'], \
                                                                    [values[0]] * len(res_df), \
                                                                    [values[1]] * len(res_df)

if interpolation_method != -1:
    if calc_u_method == 1:
        res_df['eps'] = [values[4]] * len(res_df)
    elif calc_u_method == 0:
        tmp_w = np.array(res_df['w_values'])
        tmp_q2 = np.array(res_df['q2_values'])
        tmp_ebeam = values[3]
        nu = (tmp_w ** 2 + tmp_w - mp * mp) / (2 * mp)
        res_df['eps'] = 1 / (1 + 2 * (nu ** 2 + tmp_w) / (4 * (tmp_ebeam - nu) * tmp_ebeam - tmp_w))

    if calc_u_method != -1:
        res_df['res_A'] = res_df['sigma_T'] + res_df['eps'] * res_df['sigma_L']
        res_df['d_res_A'] = ((res_df['dsigma_T'] ** 2) + ((res_df['eps'] * res_df['dsigma_L']) ** 2)) ** 0.5
        res_df['res_B'] = res_df['eps'] * res_df['sigma_TT']
        res_df['d_res_B'] = res_df['eps'] * res_df['dsigma_TT']
        res_df['res_C'] = ((2 * res_df['eps'] * (res_df['eps'] + 1)) ** 0.5) * res_df['sigma_LT']
        res_df['d_res_C'] = ((2 * res_df['eps'] * (res_df['eps'] + 1)) ** 0.5) * res_df['dsigma_LT']

    if (calc_u_method != -1) and (calc_cross_section_method == 1):
        if interpolation_method == 0:
            phi = res_df['x_axis_values'].copy() * (np.pi / 180)
            res_df['res_cross_sect'] = res_df['res_A'] + res_df['res_B'] * np.cos(2 * phi) + \
                                       res_df['res_C'] * np.cos(phi)
            res_df['d_res_cross_sect'] = (res_df['d_res_A'] ** 2 + (res_df['d_res_B'] * np.cos(2 * phi)) ** 2 +
                                          (res_df['d_res_C'] * np.cos(phi)) ** 2) ** 0.5
        else:
            phi=values[5]
            res_df['res_cross_sect'] = res_df['res_A'] + res_df['res_B'] * np.cos(2 * phi) + \
                                       res_df['res_C'] * np.cos(phi)
            res_df['d_res_cross_sect'] = (res_df['d_res_A'] ** 2 + (res_df['d_res_B'] * np.cos(2 * phi)) ** 2 +
                                          (res_df['d_res_C'] * np.cos(phi)) ** 2) ** 0.5


res_df.sort_values(by='x_axis_values', inplace=True)

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

particle_class = "Pin"
w_user = "1.5"
q2_user = "0.5"
cos_user = "0.2"
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
elif particle_class == 'Pi0P':
    PartNum = '1213'
    ParticleSecret = 'PI0P'
    ParticleBeauty = 'gvp--->ПЂвЃ°p'
    dataframe = df[(df.Channel == 9) | (df.Channel == 37) | (df.Channel == 170)].copy()

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

for el in values:
    try:
        _ = float(el)
        our_method.append(1)
    except:
        our_method.append(0)

interpolation_method = 0
calc_u_method = 0
calc_cross_section_method = 0

if our_method[0:3] == [1, 1, 1]:  # W, Q2, cos filled
    interpolation_method = 4
    x_axis_values = np.linspace(-180, 180, 100)
    x_axis_name = "phi"
elif our_method[0:3] == [1, 1, 0]:  # W, Q2 filled
    interpolation_method = 1
    x_axis_values = np.linspace(-1, 1, 100)
    col_1, col_2, col_3 = dataframe['w_average'].tolist(), dataframe['q2_average'].tolist(), dataframe[
        'Cos(theta)'].tolist()
    val_1=w_user
    val_2=q2_user
    x_axis_name = "cos(theta)"
elif our_method[0:3] == [1, 0, 1]:  # W, cos filled
    interpolation_method = 2
    x_axis_values = np.linspace(0, 6, 100)
    col_1, col_2, col_3 = dataframe['w_average'].tolist(), dataframe['Cos(theta)'].tolist(), dataframe[
        'q2_average'].tolist()
    val_1=w_user
    val_2=cos_user
    x_axis_name = "q2(gev2)"
elif our_method[0:3] == [0, 1, 1]:  # Q2, cos filled
    interpolation_method = 3
    x_axis_values = np.linspace(0, 6, 100)
    col_1, col_2, col_3 = dataframe['q2_average'].tolist(), dataframe['Cos(theta)'].tolist(), dataframe[
        'w_average'].tolist()
    val_1=q2_user
    val_2=cos_user
    x_axis_name = "w(gev)"

if interpolation_method != 0:
    if our_method[3] == 1:
        calc_u_method = 1
    elif our_method[4] == 1:
        calc_u_method = 2

if (interpolation_method != 0) and (calc_u_method != 0):
    if our_method[5] == 1:
        calc_cross_section_method = 1










# def interpolation(col_1, col_2, col_3, val_1, val_2, x_list):
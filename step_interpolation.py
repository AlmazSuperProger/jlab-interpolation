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

import maid_almaz_version

# считываем данные, введенные пользователем

gettext = cgi.FieldStorage()
mp = 0.93827

df = pd.read_csv('final_table.csv', header=None, sep='\t',
                 names=['Channel', 'MID', 'Wmin', 'Wmax', 'Q2min', 'Q2max', 'Cos(theta)', 'Sigma_T', 'dSigma_T',
                        'Sigma_L', 'dSigma_L', 'Sigma_TT', 'dSigma_TT', 'Sigma_LT', 'dSigma_LT', 'eps'])
df = df.assign(w_average=((df['Wmin'] + df['Wmax']) / 2))
df = df.assign(q2_average=((df['Q2min'] + df['Q2max']) / 2))

this_min_value = -0.6
this_max_value = 1
points_num_user = 8
points_num_checkbox = False


def interpolate_in_one_region(dataframe, particle, w_min, w_max, q2_min, q2_max, w_interp='empty', q2_interp='empty',
                              cos_interp='empty', grid_step='0.01'):
    if str(w_interp) == 'empty':
        points_num = int((w_max - w_min) / float(grid_step.replace(',', '.')))
        x_axis_values = np.linspace(w_min, w_max, points_num)
        x_axis_label = 'W(GeV)'
        interp_df = dataframe[
            (dataframe.w_average >= w_min) & (dataframe.w_average <= w_max) & (dataframe.q2_average >= q2_min) & (
                        dataframe.q2_average <= q2_max)].copy()
        sigma_TT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['Sigma_TT'], (x_axis_values, q2_interp, cos_interp), method='linear',
                            rescale=True)
        sigma_LT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['Sigma_LT'], (x_axis_values, q2_interp, cos_interp), method='linear',
                            rescale=True)
        sigma_T = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                           interp_df['Sigma_T'], (x_axis_values, q2_interp, cos_interp), method='linear', rescale=True)
        sigma_L = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                           interp_df['Sigma_L'], (x_axis_values, q2_interp, cos_interp), method='linear', rescale=True)
        dsigma_TT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                             interp_df['dSigma_TT'], (x_axis_values, q2_interp, cos_interp), method='linear',
                             rescale=True)
        dsigma_LT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                             interp_df['dSigma_LT'], (x_axis_values, q2_interp, cos_interp), method='linear',
                             rescale=True)
        dsigma_T = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['dSigma_T'], (x_axis_values, q2_interp, cos_interp), method='linear',
                            rescale=True)
        dsigma_L = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['dSigma_L'], (x_axis_values, q2_interp, cos_interp), method='linear',
                            rescale=True)
    elif str(q2_interp) == 'empty':
        points_num = int((q2_max - q2_min) / float(grid_step.replace(',', '.')))
        x_axis_values = np.linspace(q2_min, q2_max, points_num)
        x_axis_label = 'Q2 (GeV2)'
        interp_df = dataframe[
            (dataframe.w_average >= w_min) & (dataframe.w_average <= w_max) & (dataframe.q2_average >= q2_min) & (
                        dataframe.q2_average <= q2_max)].copy()
        sigma_TT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['Sigma_TT'], (w_interp, x_axis_values, cos_interp), method='linear', rescale=True)
        sigma_LT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['Sigma_LT'], (w_interp, x_axis_values, cos_interp), method='linear', rescale=True)
        sigma_T = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                           interp_df['Sigma_T'], (w_interp, x_axis_values, cos_interp), method='linear', rescale=True)
        sigma_L = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                           interp_df['Sigma_L'], (w_interp, x_axis_values, cos_interp), method='linear', rescale=True)
        dsigma_TT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                             interp_df['dSigma_TT'], (w_interp, x_axis_values, cos_interp), method='linear',
                             rescale=True)
        dsigma_LT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                             interp_df['dSigma_LT'], (w_interp, x_axis_values, cos_interp), method='linear',
                             rescale=True)
        dsigma_T = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['dSigma_T'], (w_interp, x_axis_values, cos_interp), method='linear', rescale=True)
        dsigma_L = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['dSigma_L'], (w_interp, x_axis_values, cos_interp), method='linear', rescale=True)


    elif str(cos_interp) == 'empty':
        points_num = int(2 / float(grid_step.replace(',', '.')))
        x_axis_values = np.linspace(-1, 1, points_num)
        if points_num_checkbox:
            x_axis_values = np.linspace(this_min_value, this_max_value, points_num_user)
        x_axis_label = 'Cos(theta)'
        interp_df = dataframe[
            (dataframe.w_average >= w_min) & (dataframe.w_average <= w_max) & (dataframe.q2_average >= q2_min) & (
                        dataframe.q2_average <= q2_max)].copy()
        sigma_TT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['Sigma_TT'], (w_interp, q2_interp, x_axis_values), method='linear', rescale=True)
        sigma_LT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['Sigma_LT'], (w_interp, q2_interp, x_axis_values), method='linear', rescale=True)
        sigma_T = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                           interp_df['Sigma_T'], (w_interp, q2_interp, x_axis_values), method='linear', rescale=True)
        sigma_L = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                           interp_df['Sigma_L'], (w_interp, q2_interp, x_axis_values), method='linear', rescale=True)
        dsigma_TT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                             interp_df['dSigma_TT'], (w_interp, q2_interp, x_axis_values), method='linear',
                             rescale=True)
        dsigma_LT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                             interp_df['dSigma_LT'], (w_interp, q2_interp, x_axis_values), method='linear',
                             rescale=True)
        dsigma_T = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['dSigma_T'], (w_interp, q2_interp, x_axis_values), method='linear', rescale=True)
        dsigma_L = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['dSigma_L'], (w_interp, q2_interp, x_axis_values), method='linear', rescale=True)

    elif (str(cos_interp) != 'empty') & (str(q2_interp) != 'empty') & (str(w_interp) != 'empty'):
        x_axis_label = 'phi(degree)'
        x_axis_values = np.zeros(1)
        interp_df = dataframe[
            (dataframe.w_average >= w_min) & (dataframe.w_average <= w_max) & (dataframe.q2_average >= q2_min) & (
                        dataframe.q2_average <= q2_max)].copy()
        sigma_TT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['Sigma_TT'], (w_interp, q2_interp, cos_interp), method='linear', rescale=True)
        sigma_LT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['Sigma_LT'], (w_interp, q2_interp, cos_interp), method='linear', rescale=True)
        sigma_T = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                           interp_df['Sigma_T'], (w_interp, q2_interp, cos_interp), method='linear', rescale=True)
        sigma_L = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                           interp_df['Sigma_L'], (w_interp, q2_interp, cos_interp), method='linear', rescale=True)
        dsigma_TT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                             interp_df['dSigma_TT'], (w_interp, q2_interp, cos_interp), method='linear', rescale=True)
        dsigma_LT = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                             interp_df['dSigma_LT'], (w_interp, q2_interp, cos_interp), method='linear', rescale=True)
        dsigma_T = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['dSigma_T'], (w_interp, q2_interp, cos_interp), method='linear', rescale=True)
        dsigma_L = griddata((interp_df.w_average, interp_df.q2_average, interp_df['Cos(theta)']),
                            interp_df['dSigma_L'], (w_interp, q2_interp, cos_interp), method='linear', rescale=True)

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


def append_interpolated_data(list_of_lists):
    result_list = []
    for list_array in list_of_lists:
        for elem in list_array:
            result_list.append(elem)
    return result_list


def get_full_interpolation(dataframe_join, particle_join, w_interp='empty', q2_interp='empty', cos_interp='empty',
                           grid_step='0.01'):
    x_axis_values, sigma_TT, sigma_LT, sigma_T, sigma_L, dsigma_TT, dsigma_LT, dsigma_T, dsigma_L = [], [], [], [], [], [], [], [], []
    if particle_join == 'Pin':
        x_axis_values1, sigma_TT1, sigma_LT1, sigma_T1, sigma_L1, dsigma_TT1, dsigma_LT1, dsigma_T1, dsigma_L1 = interpolate_in_one_region(
            dataframe=dataframe_join,
            particle=particle_join, w_min=1.1, w_max=1.4, q2_min=0.2, q2_max=0.7, w_interp=w_interp,
            q2_interp=q2_interp, cos_interp=cos_interp, grid_step=grid_step)
        x_axis_values2, sigma_TT2, sigma_LT2, sigma_T2, sigma_L2, dsigma_TT2, dsigma_LT2, dsigma_T2, dsigma_L2 = interpolate_in_one_region(
            dataframe=dataframe_join,
            particle=particle_join, w_min=1.4, w_max=1.6, q2_min=0.2, q2_max=0.7, w_interp=w_interp,
            q2_interp=q2_interp, cos_interp=cos_interp, grid_step=grid_step)
        x_axis_values3, sigma_TT3, sigma_LT3, sigma_T3, sigma_L3, dsigma_TT3, dsigma_LT3, dsigma_T3, dsigma_L3 = interpolate_in_one_region(
            dataframe=dataframe_join,
            particle=particle_join, w_min=1.1, w_max=1.6, q2_min=1.5, q2_max=5, w_interp=w_interp, q2_interp=q2_interp,
            cos_interp=cos_interp, grid_step=grid_step)
        x_axis_values4, sigma_TT4, sigma_LT4, sigma_T4, sigma_L4, dsigma_TT4, dsigma_LT4, dsigma_T4, dsigma_L4 = interpolate_in_one_region(
            dataframe=dataframe_join,
            particle=particle_join, w_min=1.6, w_max=2.1, q2_min=1.5, q2_max=5, w_interp=w_interp, q2_interp=q2_interp,
            cos_interp=cos_interp, grid_step=grid_step)

        x_axis_values = append_interpolated_data([x_axis_values1, x_axis_values2, x_axis_values3, x_axis_values4])
        sigma_TT = append_interpolated_data([sigma_TT1, sigma_TT2, sigma_TT3, sigma_TT4])
        sigma_LT = append_interpolated_data([sigma_LT1, sigma_LT2, sigma_LT3, sigma_LT4])
        sigma_T = append_interpolated_data([sigma_T1, sigma_T2, sigma_T3, sigma_T4])
        sigma_L = append_interpolated_data([sigma_L1, sigma_L2, sigma_L3, sigma_L4])
        dsigma_TT = append_interpolated_data([dsigma_TT1, dsigma_TT2, dsigma_TT3, dsigma_TT4])
        dsigma_LT = append_interpolated_data([dsigma_LT1, dsigma_LT2, dsigma_LT3, dsigma_LT4])
        dsigma_T = append_interpolated_data([dsigma_T1, dsigma_T2, dsigma_T3, dsigma_T4])
        dsigma_L = append_interpolated_data([dsigma_L1, dsigma_L2, dsigma_L3, dsigma_L4])

    if particle_join == 'Pi0P':
        x_axis_values1, sigma_TT1, sigma_LT1, sigma_T1, sigma_L1, dsigma_TT1, dsigma_LT1, dsigma_T1, dsigma_L1 = interpolate_in_one_region(
            dataframe=dataframe_join,
            particle=particle_join, w_min=1, w_max=1.8, q2_min=0.3, q2_max=1.9, w_interp=w_interp, q2_interp=q2_interp,
            cos_interp=cos_interp, grid_step=grid_step)
        x_axis_values2, sigma_TT2, sigma_LT2, sigma_T2, sigma_L2, dsigma_TT2, dsigma_LT2, dsigma_T2, dsigma_L2 = interpolate_in_one_region(
            dataframe=dataframe_join,
            particle=particle_join, w_min=1, w_max=4, q2_min=2.9, q2_max=6.1, w_interp=w_interp, q2_interp=q2_interp,
            cos_interp=cos_interp, grid_step=grid_step)

        x_axis_values = append_interpolated_data([x_axis_values1, x_axis_values2])
        sigma_TT = append_interpolated_data([sigma_TT1, sigma_TT2])
        sigma_LT = append_interpolated_data([sigma_LT1, sigma_LT2])
        sigma_T = append_interpolated_data([sigma_T1, sigma_T2])
        sigma_L = append_interpolated_data([sigma_L1, sigma_L2])
        dsigma_TT = append_interpolated_data([dsigma_TT1, dsigma_TT2])
        dsigma_LT = append_interpolated_data([dsigma_LT1, dsigma_LT2])
        dsigma_T = append_interpolated_data([dsigma_T1, dsigma_T2])
        dsigma_L = append_interpolated_data([dsigma_L1, dsigma_L2])

    return (x_axis_values, sigma_TT, sigma_LT, sigma_T, sigma_L, dsigma_TT, dsigma_LT, dsigma_T, dsigma_L)


class simpleMeasure(object):

    def __init__(self, w_class='empty', q2_class='empty', cos_class='empty', e_beam_class='empty', eps_class='empty',
                 phi_class='empty', particle_class='Pin', grid_step_class='0.01'):
        self.cant_calculate_eps = False
        self.method = 0
        self.x_label = '0'
        self.check_w = False
        self.check_q2 = False
        self.check_cos = False
        self.check_e_beam = False
        self.check_phi = False
        self.check_eps = False

        self.exp_x = []
        self.exp_y = [[], [], [], []]

        self.grid_step = grid_step_class

        self.particle_class = particle_class

        self.particle_class = particle_class
        self.eps_class = eps_class.replace(',', '.')
        self.w_class = w_class.replace(',', '.')
        self.q2_class = q2_class.replace(',', '.')
        self.cos_class = cos_class.replace(',', '.')
        self.e_beam_class = e_beam_class.replace(',', '.')
        self.phi_class = phi_class.replace(',', '.')

        try:
            self.w_class = float(self.w_class)
            self.check_w = True
        except:
            self.check_w = False
            self.w_class = 'empty'

        try:
            self.q2_class = float(self.q2_class)
            self.check_q2 = True
        except:
            self.check_q2 = False
            self.check_q2 = 'empty'

        try:
            self.cos_class = float(self.cos_class)
            self.check_cos = True
        except:
            self.check_cos = False
            self.cos_class = 'empty'

        try:
            self.phi_class = float(self.phi_class)
            self.check_phi = True
        except:
            self.check_phi = False
            self.phi_class = 'empty'

        try:
            self.e_beam_class = float(self.e_beam_class)
            self.check_e_beam = True
        except:
            self.check_e_beam = False
            self.e_beam_class = 'empty'

        try:
            self.eps_class = float(self.eps_class)
            self.check_eps = True
        except:
            self.check_eps = False
            self.eps_class = 'empty'

        if (self.particle_class == 'Pin'):
            self.partNum = '1212'
            self.ParticleSecret = 'PIN'
            self.ParticleBeauty = 'gvp--->ПЂвЃєn'
            self.dataframe = df[
                (df.Channel == 8) | (df.Channel == 14) | (df.Channel == 41) | (df.Channel == 141)].copy()

        if (self.particle_class == 'Pi0P'):
            self.PartNum = '1213'
            self.ParticleSecret = 'PI0P'
            self.ParticleBeauty = 'gvp--->ПЂвЃ°p'
            self.dataframe = df[(df.Channel == 9) | (df.Channel == 37) | (df.Channel == 170)].copy()

        self.x_axis_values, self.sigma_TT, self.sigma_LT, self.sigma_T, self.sigma_L, self.dsigma_TT, self.dsigma_LT, self.dsigma_T, self.dsigma_L = get_full_interpolation(
            dataframe_join=self.dataframe,
            particle_join=self.particle_class, w_interp=self.w_class, q2_interp=self.q2_class,
            cos_interp=self.cos_class, grid_step=self.grid_step)
        self.x_axis_values = np.array(self.x_axis_values)
        self.sigma_TT = np.array(self.sigma_TT)
        self.sigma_LT = np.array(self.sigma_LT)
        self.sigma_T = np.array(self.sigma_T)
        self.sigma_L = np.array(self.sigma_L)
        self.dsigma_TT = np.array(self.dsigma_TT)
        self.dsigma_LT = np.array(self.dsigma_LT)
        self.dsigma_T = np.array(self.dsigma_T)
        self.dsigma_L = np.array(self.dsigma_L)
        self.eps_class = np.full(len(self.x_axis_values), self.eps_class)

        try:
            if not (self.check_eps) and self.check_e_beam:
                self.eps_class = []
                if str(self.w_class) == 'empty':
                    mp = 0.93827
                    nu = ((self.x_axis_values ** 2) + self.q2_class - mp * mp) / (2 * mp)
                    self.eps_class = 1 / (1 + 2 * (nu * 2 + self.q2_class) / (
                                4 * (self.e_beam_class - nu) * self.e_beam_class - self.q2_class))

                elif str(self.q2_class) == 'empty':
                    mp = 0.93827
                    nu = (self.w_class ** 2 + self.x_axis_values - mp * mp) / (2 * mp)
                    self.eps_class = 1 / (1 + 2 * (nu ** 2 + self.x_axis_values) / (
                                4 * (self.e_beam_class - nu) * self.e_beam_class - self.x_axis_values))

                elif not (str(self.w_class) == 'empty') and not (str(self.q2_class) == 'empty'):
                    mp = 0.93827
                    nu = (self.w_class ** 2 + self.q2_class - mp * mp) / (2 * mp)
                    self.eps_class = 1 / (1 + 2 * (nu ** 2 + self.q2_class) / (
                                4 * (self.e_beam_class - nu) * self.e_beam_class - self.q2_class))
                    self.eps_class = np.full(len(self.x_axis_values), self.eps_class)
        except:
            self.cant_calculate_eps = True
            self.check_eps = False
            self.check_e_beam = False

        if self.check_eps or self.check_e_beam:
            self.method = 1
            self.res_A = self.sigma_T + self.eps_class * self.sigma_L
            self.d_res_A = ((self.dsigma_T ** 2) + ((self.eps_class * self.dsigma_L) ** 2)) ** 0.5
            self.res_B = self.eps_class * self.sigma_TT
            self.d_res_B = self.eps_class * self.dsigma_TT
            self.res_C = ((2 * self.eps_class * (self.eps_class + 1)) ** 0.5) * self.sigma_LT
            self.d_res_C = ((2 * self.eps_class * (self.eps_class + 1)) ** 0.5) * self.dsigma_LT

        if self.check_phi and (self.check_eps or self.check_e_beam):
            self.method = 2
            phi = self.phi_class * (np.pi / 180)
            self.res_cross_sect = self.res_A + self.res_B * np.cos(2 * phi) + self.res_C * np.cos(phi)
            self.d_res_cross_sect = (self.d_res_A ** 2 + (self.d_res_B * np.cos(2 * phi)) ** 2 + (
                        self.d_res_C * np.cos(phi)) ** 2) ** 0.5

        if self.check_phi and (self.check_eps or self.check_e_beam):
            self.res_df = pd.DataFrame({'x_axis_values': self.x_axis_values,
                                        'sigma_TT': self.sigma_TT,
                                        'sigma_LT': self.sigma_LT,
                                        'sigma_T': self.sigma_T,
                                        'sigma_L': self.sigma_L,
                                        'd_sigma_TT': self.dsigma_TT,
                                        'd_sigma_LT': self.dsigma_LT,
                                        'd_sigma_T': self.dsigma_T,
                                        'd_sigma_L': self.dsigma_L,
                                        'res_A': self.res_A,
                                        'res_B': self.res_B,
                                        'res_C': self.res_C,
                                        'd_res_A': self.d_res_A,
                                        'd_res_B': self.d_res_B,
                                        'd_res_C': self.d_res_C,
                                        'res_cross_sect': self.res_cross_sect,
                                        'd_res_cross_sect': self.d_res_cross_sect})
            self.res_df = self.res_df.sort_values(by='x_axis_values')
            self.res_cross_sect = np.array(self.res_df['res_cross_sect'])
            self.d_res_cross_sect = np.array(self.res_df['d_res_cross_sect'])

        if not self.check_phi and (self.check_eps or self.check_e_beam):
            self.res_df = pd.DataFrame({'x_axis_values': self.x_axis_values,
                                        'sigma_TT': self.sigma_TT,
                                        'sigma_LT': self.sigma_LT,
                                        'sigma_T': self.sigma_T,
                                        'sigma_L': self.sigma_L,
                                        'd_sigma_TT': self.dsigma_TT,
                                        'd_sigma_LT': self.dsigma_LT,
                                        'd_sigma_T': self.dsigma_T,
                                        'd_sigma_L': self.dsigma_L,
                                        'res_A': self.res_A,
                                        'res_B': self.res_B,
                                        'res_C': self.res_C,
                                        'd_res_A': self.d_res_A,
                                        'd_res_B': self.d_res_B,
                                        'd_res_C': self.d_res_C
                                        })
            self.res_df = self.res_df.sort_values(by='x_axis_values')
            self.x_axis_values = np.array(self.res_df['x_axis_values'])
            self.res_A = list(self.res_df['res_A'])
            self.d_res_A = list(self.res_df['d_res_A'])
            self.res_B = list(self.res_df['res_B'])
            self.res_C = list(self.res_df['res_C'])
            self.d_res_C = list(self.res_df['d_res_C'])
            self.d_res_B = list(self.res_df['d_res_B'])
            self.sigma_TT = list(self.res_df['sigma_TT'])
            self.sigma_LT = list(self.res_df['sigma_LT'])
            self.sigma_T = list(self.res_df['sigma_T'])
            self.sigma_L = list(self.res_df['sigma_L'])
            self.dsigma_TT = list(self.res_df['d_sigma_TT'])
            self.dsigma_LT = list(self.res_df['d_sigma_LT'])
            self.dsigma_T = list(self.res_df['d_sigma_T'])
            self.dsigma_L = list(self.res_df['d_sigma_L'])

        if not self.check_phi and not (self.check_eps or self.check_e_beam):
            self.res_df = pd.DataFrame({'x_axis_values': self.x_axis_values,
                                        'sigma_TT': self.sigma_TT,
                                        'sigma_LT': self.sigma_LT,
                                        'sigma_T': self.sigma_T,
                                        'sigma_L': self.sigma_L,
                                        'd_sigma_TT': self.dsigma_TT,
                                        'd_sigma_LT': self.dsigma_LT,
                                        'd_sigma_T': self.dsigma_T,
                                        'd_sigma_L': self.dsigma_L,
                                        })
            self.res_df = self.res_df.sort_values(by='x_axis_values')
            self.x_axis_values = np.array(self.res_df['x_axis_values'])
            self.sigma_TT = list(self.res_df['sigma_TT'])
            self.sigma_LT = list(self.res_df['sigma_LT'])
            self.sigma_T = list(self.res_df['sigma_T'])
            self.sigma_L = list(self.res_df['sigma_L'])
            self.dsigma_TT = list(self.res_df['d_sigma_TT'])
            self.dsigma_LT = list(self.res_df['d_sigma_LT'])
            self.dsigma_T = list(self.res_df['d_sigma_T'])
            self.dsigma_L = list(self.res_df['d_sigma_L'])


def calculate_for_structure_function(this_object):
    sort_dataframe = pd.DataFrame({'cos_theta': this_object.x_axis_values,
                                   'sigma_T': this_object.sigma_T,
                                   'sigma_L': this_object.sigma_L,
                                   'sigma_LT': this_object.sigma_LT,
                                   'sigma_TT': this_object.sigma_TT,

                                   'dsigma_T': this_object.dsigma_T,
                                   'dsigma_LT': this_object.dsigma_LT,
                                   'dsigma_L': this_object.dsigma_L,
                                   'dsigma_TT': this_object.dsigma_TT})

    sort_dataframe = sort_dataframe.sort_values(by='cos_theta')
    sort_dataframe.reset_index(drop=True, inplace=True)

    x_ax_val = np.array(sort_dataframe['cos_theta'])
    sig_T = np.array(sort_dataframe['sigma_T'])
    sig_L = np.array(sort_dataframe['sigma_L'])
    sig_TT = np.array(sort_dataframe['sigma_TT'])
    sig_LT = np.array(sort_dataframe['sigma_LT'])

    d_sig_T = np.array(sort_dataframe['dsigma_T'])
    d_sig_L = np.array(sort_dataframe['dsigma_L'])
    d_sig_TT = np.array(sort_dataframe['dsigma_TT'])
    d_sig_LT = np.array(sort_dataframe['dsigma_LT'])

    upper_T = sig_T + d_sig_T
    upper_L = sig_L + d_sig_L
    upper_TT = sig_TT + d_sig_TT
    upper_LT = sig_LT + d_sig_LT

    integ_T = 0
    integ_L = 0
    integ_TT = 0
    integ_LT = 0

    upper_integ_T = 0
    upper_integ_L = 0
    upper_integ_TT = 0
    upper_integ_LT = 0

    integ_dT = 0
    integ_dL = 0
    integ_dTT = 0
    integ_dLT = 0

    for j in range(0, len(x_ax_val) - 1):
        integ_T = integ_T + ((sig_T[j] + sig_T[j + 1])) * \
                  (x_ax_val[j + 1] - x_ax_val[j]) / 2

        integ_L = integ_L + ((sig_L[j] + sig_L[j + 1])) * \
                  (x_ax_val[j + 1] - x_ax_val[j]) / 2

        integ_TT = integ_TT + ((sig_TT[j] + sig_TT[j + 1])) * \
                   (x_ax_val[j + 1] - x_ax_val[j]) / 2

        integ_LT = integ_LT + ((sig_LT[j] + sig_LT[j + 1])) * \
                   (x_ax_val[j + 1] - x_ax_val[j]) / 2

        upper_integ_T = upper_integ_T + ((upper_T[j] + upper_T[j + 1])) * \
                        (x_ax_val[j + 1] - x_ax_val[j]) / 2

        upper_integ_L = upper_integ_L + ((upper_L[j] + upper_L[j + 1])) * \
                        (x_ax_val[j + 1] - x_ax_val[j]) / 2

        upper_integ_TT = upper_integ_TT + ((upper_TT[j] + upper_TT[j + 1])) * \
                         (x_ax_val[j + 1] - x_ax_val[j]) / 2

        upper_integ_LT = upper_integ_LT + ((upper_LT[j] + upper_LT[j + 1])) * \
                         (x_ax_val[j + 1] - x_ax_val[j]) / 2

    integ_T = integ_T * 2 * np.pi
    integ_L = integ_L * 2 * np.pi
    integ_TT = integ_TT * 2 * np.pi
    integ_LT = integ_LT * 2 * np.pi

    upper_integ_T = upper_integ_T * 2 * np.pi
    upper_integ_L = upper_integ_L * 2 * np.pi
    upper_integ_TT = upper_integ_TT * 2 * np.pi
    upper_integ_LT = upper_integ_LT * 2 * np.pi

    integ_dT = abs(upper_integ_T - integ_T)
    integ_dL = abs(upper_integ_L - integ_L)
    integ_dTT = abs(upper_integ_TT - integ_TT)
    integ_dLT = abs(upper_integ_LT - integ_LT)

    return integ_T, integ_L, integ_TT, integ_LT, integ_dT, integ_dL, integ_dTT, integ_dLT


def calculate_for_experimental_structure_function(this_object):
    sort_dataframe = pd.DataFrame({'cos_theta': this_object.exp_x,
                                   'sigma_T': this_object.exp_y[0],
                                   'sigma_L': this_object.exp_y[1],
                                   'sigma_LT': this_object.exp_y[3],
                                   'sigma_TT': this_object.exp_y[2],

                                   'dsigma_T': this_object.exp_dy[0],
                                   'dsigma_LT': this_object.exp_dy[3],
                                   'dsigma_L': this_object.exp_dy[1],
                                   'dsigma_TT': this_object.exp_dy[2]
                                   })

    sort_dataframe = sort_dataframe.sort_values(by='cos_theta')
    sort_dataframe.reset_index(drop=True, inplace=True)

    x_ax_val = np.array(sort_dataframe['cos_theta'])
    sig_T = np.array(sort_dataframe['sigma_T'])
    sig_L = np.array(sort_dataframe['sigma_L'])
    sig_TT = np.array(sort_dataframe['sigma_TT'])
    sig_LT = np.array(sort_dataframe['sigma_LT'])
    d_sig_T = np.array(sort_dataframe['dsigma_T'])
    d_sig_L = np.array(sort_dataframe['dsigma_L'])
    d_sig_TT = np.array(sort_dataframe['dsigma_TT'])
    d_sig_LT = np.array(sort_dataframe['dsigma_LT'])

    upper_T = sig_T + d_sig_T
    upper_L = sig_L + d_sig_L
    upper_TT = sig_TT + d_sig_TT
    upper_LT = sig_LT + d_sig_LT

    integ_T = 0
    integ_L = 0
    integ_TT = 0
    integ_LT = 0

    upper_integ_T = 0
    upper_integ_L = 0
    upper_integ_TT = 0
    upper_integ_LT = 0

    integ_dT = 0
    integ_dL = 0
    integ_dTT = 0
    integ_dLT = 0

    for j in range(0, len(x_ax_val) - 1):
        integ_T = integ_T + ((sig_T[j] + sig_T[j + 1])) * \
                  (x_ax_val[j + 1] - x_ax_val[j]) / 2

        integ_L = integ_L + ((sig_L[j] + sig_L[j + 1])) * \
                  (x_ax_val[j + 1] - x_ax_val[j]) / 2

        integ_TT = integ_TT + ((sig_TT[j] + sig_TT[j + 1])) * \
                   (x_ax_val[j + 1] - x_ax_val[j]) / 2

        integ_LT = integ_LT + ((sig_LT[j] + sig_LT[j + 1])) * \
                   (x_ax_val[j + 1] - x_ax_val[j]) / 2

        upper_integ_T = upper_integ_T + ((upper_T[j] + upper_T[j + 1])) * \
                        (x_ax_val[j + 1] - x_ax_val[j]) / 2

        upper_integ_L = upper_integ_L + ((upper_L[j] + upper_L[j + 1])) * \
                        (x_ax_val[j + 1] - x_ax_val[j]) / 2

        upper_integ_TT = upper_integ_TT + ((upper_TT[j] + upper_TT[j + 1])) * \
                         (x_ax_val[j + 1] - x_ax_val[j]) / 2

        upper_integ_LT = upper_integ_LT + ((upper_LT[j] + upper_LT[j + 1])) * \
                         (x_ax_val[j + 1] - x_ax_val[j]) / 2

    integ_T = integ_T * 2 * np.pi
    integ_L = integ_L * 2 * np.pi
    integ_TT = integ_TT * 2 * np.pi
    integ_LT = integ_LT * 2 * np.pi

    upper_integ_T = upper_integ_T * 2 * np.pi
    upper_integ_L = upper_integ_L * 2 * np.pi
    upper_integ_TT = upper_integ_TT * 2 * np.pi
    upper_integ_LT = upper_integ_LT * 2 * np.pi

    integ_dT = abs(upper_integ_T - integ_T)
    integ_dL = abs(upper_integ_L - integ_L)
    integ_dTT = abs(upper_integ_TT - integ_TT)
    integ_dLT = abs(upper_integ_LT - integ_LT)

    return (integ_T, integ_L, integ_TT, integ_LT, integ_dT, integ_dL, integ_dTT, integ_dLT)


def cut_structure_functions(this_object, user_choose=False, exp_data_is_available=False, this_min_val=0,
                            this_max_val=0):
    if exp_data_is_available and user_choose:
        df_to_cut = pd.DataFrame({'cos_theta': this_object.x_axis_values,
                                  'sigma_T': this_object.sigma_T,
                                  'sigma_L': this_object.sigma_L,
                                  'sigma_LT': this_object.sigma_LT,
                                  'sigma_TT': this_object.sigma_TT,

                                  'dsigma_T': this_object.dsigma_T,
                                  'dsigma_LT': this_object.dsigma_LT,
                                  'dsigma_L': this_object.dsigma_L,
                                  'dsigma_TT': this_object.dsigma_TT})
        df_to_cut = df_to_cut[(df_to_cut['cos_theta'] <= this_max_val) &
                              (df_to_cut['cos_theta'] >= this_min_val)]
        df_to_cut.reset_index(drop=True, inplace=True)
        this_object.x_axis_values = df_to_cut['cos_theta'].tolist()
        this_object.sigma_T = df_to_cut['sigma_T'].tolist()
        this_object.sigma_L = df_to_cut['sigma_L'].tolist()
        this_object.sigma_LT = df_to_cut['sigma_LT'].tolist()
        this_object.sigma_TT = df_to_cut['sigma_TT'].tolist()
        this_object.dsigma_T = df_to_cut['dsigma_T'].tolist()
        this_object.dsigma_LT = df_to_cut['dsigma_LT'].tolist()
        this_object.dsigma_L = df_to_cut['dsigma_L'].tolist()
        this_object.dsigma_TT = df_to_cut['dsigma_TT'].tolist()

    return this_object


def create_object(q2_create, w_create, cos_create, phi_create, e_beam_create, particle_create, eps_create,
                  grid_step_create='0.01'):
    check_q2 = False
    check_w = False
    check_cos = False
    objct = 0

    try:
        try_q2 = float(q2_create.replace(',', '.'))
        check_q2 = True
    except:
        check_q2 = False

    try:
        try_w = float(w_create.replace(',', '.'))
        check_w = True
    except:
        check_w = False

    try:
        try_cos = float(cos_create.replace(',', '.'))
        check_cos = True
    except:
        check_cos = False

    if check_q2 and check_w and (not check_cos):
        objct = simpleMeasure(w_class=w_create, q2_class=q2_create, cos_class='empty', e_beam_class=e_beam_create,
                              eps_class=eps_create, phi_class=phi_create, particle_class=particle_create,
                              grid_step_class=grid_step_create)
        objct.x_label = 'cos(theta)'

        w_create = float(str(w_create).replace(',', '.'))
        q2_create = float(str(q2_create).replace(',', '.'))

        exp_df = objct.dataframe[
            (objct.dataframe['w_average'] == w_create) & (objct.dataframe['q2_average'] == q2_create)].copy()

        objct.exp_x = list(exp_df['Cos(theta)'])

        objct.exp_y = [list(exp_df['Sigma_T']), list(exp_df['Sigma_L']), list(exp_df['Sigma_TT']),
                       list(exp_df['Sigma_LT'])]
        objct.exp_dy = [list(exp_df['dSigma_T']), list(exp_df['dSigma_L']), list(exp_df['dSigma_TT']),
                        list(exp_df['dSigma_LT'])]










    elif (not check_q2) and check_w and check_cos:
        objct = simpleMeasure(w_class=w_create, q2_class='empty', cos_class=cos_create, e_beam_class=e_beam_create,
                              eps_class=eps_create, phi_class=phi_create, particle_class=particle_create,
                              grid_step_class=grid_step_create)
        objct.x_label = 'q2(GeV2)'

        w_create = float(str(w_create).replace(',', '.'))
        cos_create = float(str(cos_create).replace(',', '.'))

        exp_df = objct.dataframe[
            (objct.dataframe['w_average'] == w_create) & (objct.dataframe['Cos(theta)'] == cos_create)].copy()

        objct.exp_x = list(exp_df['q2_average'])

        objct.exp_y = [list(exp_df['Sigma_T']), list(exp_df['Sigma_L']), list(exp_df['Sigma_TT']),
                       list(exp_df['Sigma_LT'])]
        objct.exp_dy = [list(exp_df['dSigma_T']), list(exp_df['dSigma_L']), list(exp_df['dSigma_TT']),
                        list(exp_df['dSigma_LT'])]








    elif check_q2 and (not check_w) and check_cos:
        objct = simpleMeasure(w_class='empty', q2_class=q2_create, cos_class=cos_create, e_beam_class=e_beam_create,
                              eps_class=eps_create, phi_class=phi_create, particle_class=particle_create,
                              grid_step_class=grid_step_create)
        objct.x_label = 'W (GeV)'

        cos_create = float(str(cos_create).replace(',', '.'))
        q2_create = float(str(q2_create).replace(',', '.'))

        exp_df = objct.dataframe[
            (objct.dataframe['Cos(theta)'] == cos_create) & (objct.dataframe['q2_average'] == q2_create)].copy()

        objct.exp_x = list(exp_df['w_average'])

        objct.exp_y = [list(exp_df['Sigma_T']), list(exp_df['Sigma_L']), list(exp_df['Sigma_TT']),
                       list(exp_df['Sigma_LT'])]
        objct.exp_dy = [list(exp_df['dSigma_T']), list(exp_df['dSigma_L']), list(exp_df['dSigma_TT']),
                        list(exp_df['dSigma_LT'])]


    else:
        objct = 'empty'

    return objct


gettext = cgi.FieldStorage()

cut_data_to_exp = True if gettext.getfirst("cut_data_to_exp", "off") == "on" else False
q2_user = gettext.getfirst("q2", "empty")
w_user = gettext.getfirst("w", "empty")
cos_user1 = gettext.getfirst("cos", "empty")
phi_user = gettext.getfirst("phi", "empty")
e_beam_user = gettext.getfirst("eBeam", "empty")
particle_user = gettext.getfirst("particle", "empty")
grid_step_user = gettext.getfirst("grid_step_user", "empty")
eps_user = gettext.getfirst("eps", "empty")

this_min_value = gettext.getfirst("this_min_value", "-1")
this_max_value = gettext.getfirst("this_max_value", "1")
points_num_user = gettext.getfirst("points_num_user", "10")
points_num_checkbox = True if gettext.getfirst("points_num_checkbox", "off") == "on" else False

try:
    this_min_value = float(this_min_value.replace(',', '.'))
    this_max_value = float(this_max_value.replace(',', '.'))
    points_num_user = int(points_num_user.replace(',', '.'))

except:
    points_num_checkbox = False

try:
    gr_step = float(grid_step_user.replace(',', '.'))
except:
    grid_step_user = '0.01'

check_e_beam = False
check_phi = False
check_eps = False

try:
    try_e_beam = float(e_beam_user.replace(',', '.'))
    check_e_beam = True
except:
    check_e_beam = False

try:
    try_phi = float(phi_user.replace(',', '.'))
    check_phi = True
except:
    check_phi = False

try:
    check_eps_value = float(eps_user.replace(',', '.'))
    check_eps = True
except:
    check_eps = False

graphObj1 = create_object(q2_create=q2_user, w_create=w_user, cos_create=cos_user1,
                          phi_create=phi_user, e_beam_create=e_beam_user,
                          particle_create=particle_user, eps_create=eps_user, grid_step_create=grid_step_user)

if cut_data_to_exp:
    graphObj1 = cut_structure_functions(this_object=graphObj1,
                                        user_choose=True,
                                        exp_data_is_available=len(graphObj1.exp_x) > 1,
                                        this_min_val=min(graphObj1.exp_x),
                                        this_max_val=max(graphObj1.exp_x))

integra_values = calculate_for_structure_function(this_object=graphObj1)

experimental_integral_values = calculate_for_experimental_structure_function(graphObj1)

# possible channels: 'pi+ n', 'pi0 p', 'pi- p', 'pi0 n'

maid_particle = 'pi+ n' if particle_user == 'Pin' else 'pi0 p'
interpolator = maid_almaz_version.InterpSigmaCorrectedCached('maid', maid_particle)


def maid_structure_func(W, Q2, cos_theta):
    R_T, R_L, R_TT, R_TL, R_TLp = np.ravel(interpolator.interp_R(W, Q2, cos_theta))
    ds_T, ds_L, ds_TT, ds_TL, ds_TLp = np.ravel(interpolator.interp_dsigma_comps(W, Q2, cos_theta))
    return (R_T, R_L, R_TT, R_TL, R_TLp, ds_T, ds_L, ds_TT, ds_TL, ds_TLp)


def maid_cross_section(W, Q2, cos_theta, phi, Eb, h):
    DCS = interpolator.interp_dsigma(W, Q2, cos_theta, phi, Eb, h)
    return DCS


load_df = pd.DataFrame({'sigma_TT': graphObj1.sigma_TT,
                        'sigma_LT': graphObj1.sigma_LT,
                        'sigma_T': graphObj1.sigma_T,
                        'sigma_L': graphObj1.sigma_L,
                        'd_sigma_TT': graphObj1.dsigma_TT,
                        'd_sigma_LT': graphObj1.dsigma_LT,
                        'd_sigma_T': graphObj1.dsigma_T,
                        'd_sigma_L': graphObj1.dsigma_L})

filename_construct = '/home/almaz/public_html/save_folder/structure_functions/'
this_file_name = 'error.csv'
get_file_name = 'https://clas.sinp.msu.ru/~almaz/save_folder/structure_functions/'

if graphObj1.x_label == 'cos(theta)' and len(graphObj1.x_axis_values) > 0:
    this_file_name = 'structure_functions_q2_' + str(q2_user) + '_w_' + str(w_user) + '_' + particle_user + '.csv'
    load_df['cos(theta)'] = list(graphObj1.x_axis_values)
    load_df['w'] = list([w_user] * len(graphObj1.x_axis_values))
    load_df['q2'] = list([q2_user] * len(graphObj1.x_axis_values))

if graphObj1.x_label == 'q2(GeV2)' and len(graphObj1.x_axis_values) > 0:
    this_file_name = 'structure_functions_w_' + str(w_user) + '_cos_' + str(cos_user1) + '_' + particle_user + '.csv'
    load_df['cos(theta)'] = list([cos_user1] * len(graphObj1.x_axis_values))
    load_df['w'] = list([w_user] * len(graphObj1.x_axis_values))
    load_df['q2'] = list(graphObj1.x_axis_values)

if graphObj1.x_label == 'W (GeV)' and len(graphObj1.x_axis_values) > 0:
    this_file_name = 'structure_functions_q2_' + str(q2_user) + '_cos_' + str(cos_user1) + '_' + particle_user + '.csv'
    load_df['cos(theta)'] = list([cos_user1] * len(graphObj1.x_axis_values))
    load_df['w'] = list(graphObj1.x_axis_values)
    load_df['q2'] = list([q2_user] * len(graphObj1.x_axis_values))

new_clmn_names = ['cos(theta)', 'w', 'q2', 'sigma_TT', 'sigma_LT', 'sigma_T', 'sigma_L', 'd_sigma_TT', 'd_sigma_LT',
                  'd_sigma_T', 'd_sigma_L', ]
load_df = load_df[new_clmn_names]
filename_construct = filename_construct + this_file_name
get_file_name = get_file_name + this_file_name
load_df.to_csv(filename_construct, index=False)

maid_crushed = False

if True:
    DCS_list = []
    R_T_list, R_L_list, R_TT_list, R_TL_list, R_TLp_list, ds_T_list, ds_L_list, ds_TT_list, ds_TL_list, ds_TLp_list = [], [], [], [], [], [], [], [], [], []
    if graphObj1.x_label == 'cos(theta)':
        for i in range(0, len(graphObj1.x_axis_values)):
            this_R_T, this_R_L, this_R_TT, this_R_TL, this_R_TLp, this_ds_T, this_ds_L, this_ds_TT, this_ds_TL, this_ds_TLp = maid_structure_func(
                W=graphObj1.w_class, Q2=graphObj1.q2_class, cos_theta=graphObj1.x_axis_values[i])
            R_T_list.append(this_R_T)
            R_L_list.append(this_R_L)
            R_TT_list.append(this_R_TT)
            R_TL_list.append(this_R_TL)
            R_TLp_list.append(this_R_TLp)
            ds_T_list.append(this_ds_T)
            ds_L_list.append(this_ds_L)
            ds_TT_list.append(this_ds_TT)
            ds_TL_list.append(this_ds_TL)
            ds_TLp_list.append(this_ds_TLp)

        if graphObj1.check_e_beam and graphObj1.check_phi:
            try:
                for i in range(0, len(graphObj1.x_axis_values)):
                    DCS_this = maid_cross_section(graphObj1.w_class, graphObj1.q2_class, graphObj1.x_axis_values[i],
                                                  graphObj1.phi_class, graphObj1.e_beam_class, h=1)
                    DCS_list.append(DCS_this)
            except:
                maid_crushed = True

    if graphObj1.x_label == 'q2(GeV2)':
        for i in range(0, len(graphObj1.x_axis_values)):
            this_R_T, this_R_L, this_R_TT, this_R_TL, this_R_TLp, this_ds_T, this_ds_L, this_ds_TT, this_ds_TL, this_ds_TLp = maid_structure_func(
                W=graphObj1.w_class, Q2=graphObj1.x_axis_values[i], cos_theta=graphObj1.cos_class)
            R_T_list.append(this_R_T)
            R_L_list.append(this_R_L)
            R_TT_list.append(this_R_TT)
            R_TL_list.append(this_R_TL)
            R_TLp_list.append(this_R_TLp)
            ds_T_list.append(this_ds_T)
            ds_L_list.append(this_ds_L)
            ds_TT_list.append(this_ds_TT)
            ds_TL_list.append(this_ds_TL)
            ds_TLp_list.append(this_ds_TLp)

        if graphObj1.check_e_beam and graphObj1.check_phi:
            try:
                for i in range(0, len(graphObj1.x_axis_values)):
                    DCS_this = maid_cross_section(graphObj1.w_class, graphObj1.x_axis_values[i], graphObj1.cos_class,
                                                  graphObj1.phi_class, graphObj1.e_beam_class, h=1)
                    DCS_list.append(DCS_this)
            except:
                maid_crushed = True

    if graphObj1.x_label == 'W (GeV)':
        for i in range(0, len(graphObj1.x_axis_values)):
            this_R_T, this_R_L, this_R_TT, this_R_TL, this_R_TLp, this_ds_T, this_ds_L, this_ds_TT, this_ds_TL, this_ds_TLp = maid_structure_func(
                W=graphObj1.x_axis_values[i], Q2=graphObj1.q2_class, cos_theta=graphObj1.cos_class)
            R_T_list.append(this_R_T)
            R_L_list.append(this_R_L)
            R_TT_list.append(this_R_TT)
            R_TL_list.append(this_R_TL)
            R_TLp_list.append(this_R_TLp)
            ds_T_list.append(this_ds_T)
            ds_L_list.append(this_ds_L)
            ds_TT_list.append(this_ds_TT)
            ds_TL_list.append(this_ds_TL)
            ds_TLp_list.append(this_ds_TLp)

        if graphObj1.check_e_beam and graphObj1.check_phi:
            try:
                for i in range(0, len(graphObj1.x_axis_values)):
                    DCS_this = maid_cross_section(graphObj1.x_axis_values[i], graphObj1.q2_class, graphObj1.cos_class,
                                                  graphObj1.phi_class, graphObj1.e_beam_class, h=1)
                    DCS_list.append(DCS_this)
            except:
                maid_crushed = True


def onePlotlyGraphWithErrors(x_array, y_array, d_y_array, layout_title, x_label):
    trace = go.Scatter(
        x=x_array,
        y=y_array,
        error_y=dict(
            type='data',
            array=d_y_array,
            color='rgba(100, 100, 255, 0.6)',
            thickness=1.5,
            width=3),
        name='Interpolation',
        marker_size=1)

    data = [trace]

    fig = go.Figure(data=data)
    fig.layout.height = 700
    fig.layout.width = 1000
    fig.layout.title = layout_title

    fig.layout.yaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.5,
        title=layout_title,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))
    fig.layout.xaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.2,
        title=x_label,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))

    return fig


def onePlotlyGraphWithErrors_plus_maid(x_array, y_array, maid_y_array, d_y_array, x_exp_data, y_exp_data, dy_exp_data,
                                       layout_title, x_label):
    trace = go.Scatter(
        x=x_array,
        y=y_array,
        error_y=dict(
            type='data',
            array=d_y_array,
            color='rgba(100, 100, 255, 0.6)',
            thickness=1.5,
            width=3),
        name='Interpolation',
        marker_size=1)

    trace_maid = go.Scatter(
        x=x_array,
        y=maid_y_array,
        name='maid 2007',
        marker_size=3)

    trace_exp = go.Scatter(
        mode='markers',
        x=x_exp_data,
        y=y_exp_data,
        name='Experiment',
        marker=dict(color='rgba(100, 100, 100, 1)',
                    symbol='square'),
        error_y=dict(
            type='data',
            array=dy_exp_data,
            color='rgba(100, 100, 100, 1)',
            thickness=1.5,
            width=3),

        marker_size=10)

    data = [trace, trace_maid, trace_exp]

    fig = go.Figure(data=data)
    fig.layout.height = 700
    fig.layout.width = 1000
    fig.layout.title = layout_title

    fig.layout.yaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.5,
        title=layout_title,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))
    fig.layout.xaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.2,
        title=x_label,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))

    return fig


print("Content-type: text/html\n")
print("""<!DOCTYPE HTML>
                <html>
                <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style type="text/css">
                        A { text-decoration: none;  color: red; } 
                        * { margin: 0;}
            .textBox { width: 1440px; height:80px; margin:auto; }
            .imagesBox{ width: 1440px; height:900px; margin:auto; }
            .textBox2 { width: 1440px; height:50px; margin:auto; }
            .tableBox1 {margin:auto;  width: 1440px; height:350px;}
            .checkbox_msg { color: blue; }
            td { text-align: center ;}


                        </style>
                <meta charset="utf-8">
            <meta name="viewport" content="width=device-width">
                <script type="text/javascript"
                src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
                </script>                   
                <title>CLAS graph</title>
                    </head>
        <body>  
        <center>
    
        <br> 
    
        <br> 	<br>
	<br>
	<a href="https://clas.sinp.msu.ru/cgi-bin/almaz/instruction">Available data areas</a>
	<br>
	<br>


On this site you shoud to enter W+Q2 or W+cos(theta) or Q2+cos(theta) to get structure functions as functions of cos(theta) or Q2 or W (correspondingly).

<br>

If you also want to get unpolarized structure function you also need to enter the beam energy (OR eps).

<br>
If you also want to get differential cross section you also need to enter the beam energy (OR eps) and phi.
<br>
Grid step is an optional feature which allow you to change graphic resolution and interpolation step (default value = 0.01)
   
        
        <br><br>

""")

particle_text_1 = """<option value="Pin">gvp--->π⁺n</option>
            <option value="Pi0P">gvp--->π⁰p</option>"""

particle_text_2 = """<option value="Pi0P">gvp--->π⁰p</option>
                <option value="Pin">gvp--->π⁺n</option>"""

particle_text = particle_text_1 if particle_user == 'Pin' else particle_text_2

print("""<form method="GET" action="https://clas.sinp.msu.ru/cgi-bin/almaz/step_interpolation.py" >

      <p> <input  type="text" name="eBeam"  placeholder="Ebeam(GeV)" value="{}" > 
          <input  type="text" name="eps"  placeholder="eps" value="{}"> </p>
      <br>
      <input  type="text" name="w"  placeholder="W(GeV)" value="{}"  >
          <input  type="text" name="q2"  placeholder="Q2(GeV2)" value="{}" >
          <input  type="text" name="cos"  placeholder="Cos(theta)" value="{}"  >
          <input  type="text" name="phi"  placeholder="phi(degree)" value="{}" >
          <input  type="text" name="grid_step_user"  placeholder="grid step" value="{}"  >
          <br>
          <br>
          <input type="checkbox" id="coding" name="cut_data_to_exp" >
          <label class="checkbox_msg" for="coding">trim interpolation data (structure_functions) to a range of experimental data (if exp data is available) </label>
          
          
          
     
          <br>
          <br>
          <br>
          <br>
          <input  type="text" name="this_min_value"  placeholder="cos(theta) min value"  >
          <input  type="text" name="this_max_value"  placeholder="cos(theta) max value"  >
          <input  type="text" name="points_num_user"  placeholder="interpolation points cnt"  >
          <br>
          <br>
          <input type="checkbox" id="coding" name="points_num_checkbox" >
          <label class="checkbox_msg" for="coding">interpolate in defined cos(theta) range </label>
          
          <br>
          <br>
            <select class="select" name="particle" size="1" >
            {}
            </select>
          <br>
          <br>
          <br>
         <p> <input class="button" class="submitbutton" type="submit" value="Run">  </p>
         <br>
        </form>""".format(e_beam_user, eps_user, w_user, q2_user, cos_user1, phi_user, grid_step_user, particle_text))

print("""q2 = """, q2_user, ' GeV2<br>',
      """w = """, w_user, ' GeV <br>',
      """cos(theta) = """, cos_user1, '<br>',
      '''phi = ''', phi_user, '<br>',
      '''e_beam = ''', e_beam_user, '<br>',
      '''eps = ''', eps_user, '<br>',
      '''reaction channel: ''', particle_user, '<br>',
      '''grid step = ''', grid_step_user)

if graphObj1.cant_calculate_eps:
    print(
        '<br><br><br> ERROR: wrong E_beam for this (W,Q2,cos_theta), try to enter eps instead of the E_beam   <br><br><br>')

if maid_crushed:
    print('<br><br> maid crush <br><br>')

if True:
    if (graphObj1.check_eps or graphObj1.check_e_beam) and graphObj1.check_phi:
        if graphObj1.check_e_beam:
            fig = onePlotlyGraphWithErrors_plus_maid(x_array=graphObj1.x_axis_values,
                                                     y_array=graphObj1.res_cross_sect,
                                                     maid_y_array=DCS_list,
                                                     d_y_array=graphObj1.d_res_cross_sect,
                                                     x_exp_data=[],
                                                     y_exp_data=[],
                                                     layout_title='Diffrential cross section (mcbn/sterad)',
                                                     x_label=graphObj1.x_label)
        else:
            fig = onePlotlyGraphWithErrors(x_array=graphObj1.x_axis_values,
                                           y_array=graphObj1.res_cross_sect,
                                           d_y_array=graphObj1.d_res_cross_sect,
                                           layout_title='Diffrential cross section (mcbn/sterad)',
                                           x_label=graphObj1.x_label)
        print("{}".format(fig.to_html(full_html=False)))

    if (graphObj1.check_eps or graphObj1.check_e_beam):
        fig = onePlotlyGraphWithErrors(x_array=graphObj1.x_axis_values,
                                       y_array=graphObj1.res_A,
                                       d_y_array=graphObj1.d_res_A,
                                       layout_title='Unpolarized structure function (mcbn/sterad)',
                                       x_label=graphObj1.x_label)
        print("{}".format(fig.to_html(full_html=False)))

print("""<br> <a href="{}" download="{}">Download Your
        CSV file here</a> <br>""".format(get_file_name, this_file_name))

# print(filename_construct)

# print("<br><br>")
# print(this_file_name)

if q2_user != 'empty' and w_user != 'empty':
    link_w_value = str(w_user)
    link_q_value = str(q2_user)

    link_to_integrated_q2 = '''https://clas.sinp.msu.ru/cgi-bin/almaz/integrated_struct_function_interpolation.py?q2_user_value=''' + str(
        link_q_value) + '''&w_user_value=&reaction_channel=''' + particle_user

    link_to_integrated_w = '''https://clas.sinp.msu.ru/cgi-bin/almaz/integrated_struct_function_interpolation.py?q2_user_value=&w_user_value=''' + str(
        link_w_value) + '''&reaction_channel=''' + particle_user

    print("""<br> <a href="{}">Integrated structure function W = {} (GeV) </a> <br>""".format(link_to_integrated_w,
                                                                                              link_w_value))
    print(""" <a href="{}">Integrated structure function Q2 = {} (GeV2) </a> <br>""".format(link_to_integrated_q2,
                                                                                            link_q_value))

if True:

    if cut_data_to_exp and len(graphObj1.exp_x) > 1:
        print(""" <label class="checkbox_msg"><br><br><br>Interpolation: in experimental data range </label>""")

    else:
        print(""" <br><br><br>Interpolation:
                  <br> min interpolation cos(theta) = {}
                  <br>max interpolation cos(theta) = {}
                  <br> interpolation points cnt ={} """.format(round(min(graphObj1.x_axis_values), 3),
                                                               round(max(graphObj1.x_axis_values), 3),
                                                               len(graphObj1.x_axis_values)))

    if integra_values[0] != 0:
        print(""" <br><br><table border=2 width=800>
                <tr>
                <td>Integrated sigma_T        </td>
                <td>Integrated sigma_L        </td>
                <td>Integrated sigma_TT       </td>
                <td>Integrated sigma_LT       </td>
                </tr>
                
                <tr>
                <td>{} +- {}</td>
                <td>{} +- {}</td>
                <td>{} +- {}</td>
                <td>{} +- {}</td>
                </tr></table>""".format(round(integra_values[0], 4),
                                        round(integra_values[4], 4),
                                        round(integra_values[1], 4),
                                        round(integra_values[5], 4),
                                        round(integra_values[2], 4),
                                        round(integra_values[6], 4),
                                        round(integra_values[3], 4),
                                        round(integra_values[7], 4)))

    if experimental_integral_values[0] != 0:
        print(""" <br><br><br> Experiment:
                  <br> min experimental cos(theta) = {}
                  <br>max experimental cos(theta) = {}
                  <br> experimental points cnt ={} """.format(round(min(graphObj1.exp_x), 3),
                                                              round(max(graphObj1.exp_x), 3),
                                                              round(len(graphObj1.exp_x), 3)))
        print("""<br><br><table border=2 width=800>
                <tr>
                <td>Integrated sigma_T        </td>
                <td>Integrated sigma_L        </td>
                <td>Integrated sigma_TT       </td>
                <td>Integrated sigma_LT       </td>
                </tr>
                
                <tr>
                <td>{} +- {}</td>
                <td>{} +- {}</td>
                <td>{} +- {}</td>
                <td>{} +- {}</td>
                </tr></table>""".format(round(experimental_integral_values[0], 4),
                                        round(experimental_integral_values[4], 4),
                                        round(experimental_integral_values[1], 4),
                                        round(experimental_integral_values[5], 4),
                                        round(experimental_integral_values[2], 4),
                                        round(experimental_integral_values[6], 4),
                                        round(experimental_integral_values[3], 4),
                                        round(experimental_integral_values[7], 4)))

    fig = onePlotlyGraphWithErrors_plus_maid(x_array=graphObj1.x_axis_values,
                                             y_array=graphObj1.sigma_T,
                                             d_y_array=graphObj1.dsigma_T,
                                             maid_y_array=ds_T_list,
                                             x_exp_data=graphObj1.exp_x,
                                             y_exp_data=graphObj1.exp_y[0],
                                             dy_exp_data=graphObj1.exp_dy[0],
                                             layout_title='Transverse structure function (mcbn/sterad)',
                                             x_label=graphObj1.x_label)
    print("{}".format(fig.to_html(full_html=False)))
    fig = onePlotlyGraphWithErrors_plus_maid(x_array=graphObj1.x_axis_values,
                                             y_array=graphObj1.sigma_L,
                                             d_y_array=graphObj1.dsigma_L,
                                             maid_y_array=ds_L_list,
                                             x_exp_data=graphObj1.exp_x,
                                             y_exp_data=graphObj1.exp_y[1],
                                             dy_exp_data=graphObj1.exp_dy[1],
                                             layout_title='Longitudinal structure function (mcbn/sterad)',
                                             x_label=graphObj1.x_label)
    print("{}".format(fig.to_html(full_html=False)))
    fig = onePlotlyGraphWithErrors_plus_maid(x_array=graphObj1.x_axis_values,
                                             y_array=graphObj1.sigma_TT,
                                             maid_y_array=ds_TT_list,
                                             d_y_array=graphObj1.dsigma_TT,
                                             x_exp_data=graphObj1.exp_x,
                                             y_exp_data=graphObj1.exp_y[2],
                                             dy_exp_data=graphObj1.exp_dy[2],
                                             layout_title='Transverse-Transverse structure function (mcbn/sterad)',
                                             x_label=graphObj1.x_label)
    print("{}".format(fig.to_html(full_html=False)))
    fig = onePlotlyGraphWithErrors_plus_maid(x_array=graphObj1.x_axis_values,
                                             y_array=graphObj1.sigma_LT,
                                             d_y_array=graphObj1.dsigma_LT,
                                             maid_y_array=ds_TL_list,
                                             x_exp_data=graphObj1.exp_x,
                                             y_exp_data=graphObj1.exp_y[3],
                                             dy_exp_data=graphObj1.exp_dy[3],
                                             layout_title='Longitudinal-Transverse structure function (mcbn/sterad)',
                                             x_label=graphObj1.x_label)
    print("{}".format(fig.to_html(full_html=False)))

if False:
    fig = onePlotlyGraphWithErrors(x_array=graphObj1.x_axis_values,
                                   y_array=graphObj1.sigma_T,
                                   d_y_array=graphObj1.dsigma_T,
                                   layout_title='Transverse structure function (mcbn/sterad)',
                                   x_label=graphObj1.x_label)
    print("{}".format(fig.to_html(full_html=False)))
    fig = onePlotlyGraphWithErrors(x_array=graphObj1.x_axis_values,
                                   y_array=graphObj1.sigma_L,
                                   d_y_array=graphObj1.dsigma_L,
                                   layout_title='Longitudinal structure function (mcbn/sterad)',
                                   x_label=graphObj1.x_label)
    print("{}".format(fig.to_html(full_html=False)))
    fig = onePlotlyGraphWithErrors(x_array=graphObj1.x_axis_values,
                                   y_array=graphObj1.sigma_TT,
                                   d_y_array=graphObj1.dsigma_TT,
                                   layout_title='Transverse-Transverse structure function (mcbn/sterad)',
                                   x_label=graphObj1.x_label)
    print("{}".format(fig.to_html(full_html=False)))
    fig = onePlotlyGraphWithErrors(x_array=graphObj1.x_axis_values,
                                   y_array=graphObj1.sigma_LT,
                                   d_y_array=graphObj1.dsigma_LT,
                                   layout_title='Longitudinal-Transverse structure function (mcbn/sterad)',
                                   x_label=graphObj1.x_label)
    print("{}".format(fig.to_html(full_html=False)))

print("""</center>
        </body>
         </html>""")

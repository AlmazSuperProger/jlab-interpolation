#!/usr/bin/python3

import numpy as np
import sys
sys.path += (
    '/home/chesn/prj/clas/clasfw',
    '/home/chesn/prj/clas/evgen',
)

from sig_interpolate import InterpSigmaCorrectedCached

#
# Initialization. May take a while, should be called once at the program start
# possible channels: 'pi+ n', 'pi0 p', 'pi- p', 'pi0 n'



W  = 1.5   #  GeV
Q2 = 1.0   #  GeV^2
Eb = 10.6  #  GeV
cos_theta = 1
phi = 0    #  rad
h = 1      #  incident electron helicity



def calculate_maid_structure_func(maid_particle, maid_dataframe):
    maid_t, maid_l, maid_tt, maid_tl, d_maid_t, d_maid_l, d_maid_tt, d_maid_tl = [], [], [], [], [], [], [], []
    maid_particle = 'pi+ n' if maid_particle == 'Pin' else 'pi0 p'
    interpolator = InterpSigmaCorrectedCached('maid', maid_particle)
    for idx in maid_dataframe['w_values'].index:
        W = maid_dataframe.loc[idx, 'w_values']
        Q2 = maid_dataframe.loc[idx, 'q2_values']
        cos_theta = maid_dataframe.loc[idx, 'cos_values']
        R_T, R_L, R_TT, R_TL, R_TLp = np.ravel(interpolator.interp_R(W, Q2, cos_theta))
        ds_T, ds_L, ds_TT, ds_TL, ds_TLp = np.ravel(interpolator.interp_dsigma_comps(W, Q2, cos_theta))
        maid_t.append(R_T)
        maid_l.append(R_L)
        maid_tt.append(R_TT)
        maid_tl.append(R_TL)
        d_maid_t.append(ds_T)
        d_maid_l.append(ds_L)
        d_maid_tt.append(ds_TT)
        d_maid_tl.append(ds_TL)
    return maid_t, maid_l, maid_tt, maid_tl, d_maid_t, d_maid_l, d_maid_tt, d_maid_tl


# maid_structure_func(W,Q2,cos_theta)

# def maid_cross_section(W, Q2, cos_theta, phi, Eb, h):
#      ## differential cross-section
#     DCS = interpolator.interp_dsigma(W, Q2, cos_theta, phi, Eb, h)


#print(R_T, R_L, R_TT, R_TL, R_TLp)
#print(ds_T, ds_L, ds_TT, ds_TL, ds_TLp)
#print(DCS)

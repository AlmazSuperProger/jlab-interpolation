import numpy as np
import pandas as pd


def calculate_integral(df_to_int):
    df_to_int = df_to_int.sort_values(by='x_axis_values')
    x_ax_val = np.array(df_to_int['x_axis_values'])
    sig_t = np.array(df_to_int['sigma_t'])
    sig_l = np.array(df_to_int['sigma_l'])
    sig_tt = np.array(df_to_int['sigma_tt'])
    sig_lt = np.array(df_to_int['sigma_lt'])
    d_sig_t = np.array(df_to_int['d_sigma_t'])
    d_sig_l = np.array(df_to_int['d_sigma_l'])
    d_sig_tt = np.array(df_to_int['d_sigma_tt'])
    d_sig_lt = np.array(df_to_int['d_sigma_lt'])

    upper_t = sig_t + d_sig_t
    upper_l = sig_l + d_sig_l
    upper_tt = sig_tt + d_sig_tt
    upper_lt = sig_lt + d_sig_lt

    integ_t = 0
    integ_l = 0
    integ_tt = 0
    integ_lt = 0

    upper_integ_t = 0
    upper_integ_l = 0
    upper_integ_tt = 0
    upper_integ_lt = 0

    for j in range(0, len(x_ax_val) - 1):
        integ_t = integ_t + (sig_t[j] + sig_t[j + 1]) * (x_ax_val[j + 1] - x_ax_val[j]) / 2
        integ_l = integ_l + (sig_l[j] + sig_l[j + 1]) * (x_ax_val[j + 1] - x_ax_val[j]) / 2
        integ_tt = integ_tt + (sig_tt[j] + sig_tt[j + 1]) * (x_ax_val[j + 1] - x_ax_val[j]) / 2
        integ_lt = integ_lt + (sig_lt[j] + sig_lt[j + 1]) * (x_ax_val[j + 1] - x_ax_val[j]) / 2
        upper_integ_t = upper_integ_t + (upper_t[j] + upper_t[j + 1]) * (x_ax_val[j + 1] - x_ax_val[j]) / 2
        upper_integ_l = upper_integ_l + (upper_l[j] + upper_l[j + 1]) * (x_ax_val[j + 1] - x_ax_val[j]) / 2
        upper_integ_tt = upper_integ_tt + (upper_tt[j] + upper_tt[j + 1]) * (x_ax_val[j + 1] - x_ax_val[j]) / 2
        upper_integ_lt = upper_integ_lt + (upper_lt[j] + upper_lt[j + 1]) * (x_ax_val[j + 1] - x_ax_val[j]) / 2

    integ_t = integ_t * 2 * np.pi
    integ_l = integ_l * 2 * np.pi
    integ_tt = integ_tt * 2 * np.pi
    integ_lt = integ_lt * 2 * np.pi

    upper_integ_t = upper_integ_t * 2 * np.pi
    upper_integ_l = upper_integ_l * 2 * np.pi
    upper_integ_tt = upper_integ_tt * 2 * np.pi
    upper_integ_lt = upper_integ_lt * 2 * np.pi

    integ_dt = abs(upper_integ_t - integ_t)
    integ_dl = abs(upper_integ_l - integ_l)
    integ_dtt = abs(upper_integ_tt - integ_tt)
    integ_dlt = abs(upper_integ_lt - integ_lt)

    return integ_t, integ_l, integ_tt, integ_lt, integ_dt, integ_dl, integ_dtt, integ_dlt

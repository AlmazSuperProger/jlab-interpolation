import numpy as np
import pandas as pd
from scipy.interpolate import griddata

mp = 0.93827


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


def make_interpolation(df, particle_class_ext, w_user_ext, q2_user_ext, cos_user_ext,
                       e_beam_user_ext, eps_user_ext, phi_user_ext, interp_step_user_ext,
                       x_axis_min_ext, x_axis_max_ext):
    particle_class = particle_class_ext
    w_user = w_user_ext
    q2_user = q2_user_ext
    cos_user = cos_user_ext
    e_beam_user = e_beam_user_ext
    eps_user = eps_user_ext
    phi_user = phi_user_ext
    interp_step_user = float(interp_step_user_ext)
    x_axis_min = x_axis_min_ext
    x_axis_max = x_axis_max_ext
    x_axis_label = 'empty'

    if particle_class == 'Pin':
        partNum = '1212'
        ParticleSecret = 'PIN'
        ParticleBeauty = 'gvp->π⁺n'
        dataframe = df[
            (df.Channel == 8) | (df.Channel == 14) | (df.Channel == 41) | (df.Channel == 141)].copy()
        dataframes = [
            dataframe[(dataframe['w_average'] >= 1.1) & (dataframe['w_average'] <= 1.6) &
                      (dataframe['q2_average'] >= 0.2) & (dataframe['q2_average'] <= 0.7)],
            dataframe[(dataframe['w_average'] >= 1.1) & (dataframe['w_average'] <= 1.15) &
                      (dataframe['q2_average'] >= 2.115) & (dataframe['q2_average'] <= 4.155)],
            dataframe[(dataframe['w_average'] >= 1.15) & (dataframe['w_average'] <= 1.69) &
                      (dataframe['q2_average'] >= 1.72) & (dataframe['q2_average'] <= 4.16)],
            dataframe[(dataframe['w_average'] >= 1.605) & (dataframe['w_average'] <= 2.01) &
                      (dataframe['q2_average'] >= 1.8) & (dataframe['q2_average'] <= 4)]
        ]
        # dataframes = [
        #     dataframe[df.Channel == 8],
        #     dataframe[df.Channel == 14],
        #     dataframe[df.Channel == 41],
        #     dataframe[df.Channel == 141]]

    elif particle_class == 'Pi0P':
        PartNum = '1213'
        ParticleSecret = 'PI0P'
        ParticleBeauty = 'gvp->π⁰p'
        dataframe = df[(df.Channel == 9) | (df.Channel == 37) | (df.Channel == 170)].copy()
        dataframes = [
            dataframe[(dataframe['w_average'] >= 1) & (dataframe['w_average'] <= 1.8) &
                      (dataframe['q2_average'] >= 0.3) & (dataframe['q2_average'] <= 1.9)],
            dataframe[(dataframe['w_average'] >= 1) & (dataframe['w_average'] <= 1.4) &
                      (dataframe['q2_average'] >= 2.9) & (dataframe['q2_average'] <= 6.1)]
        ]
        # dataframes = [
        #     dataframe[df.Channel == 9],
        #     dataframe[df.Channel == 37],
        #     dataframe[df.Channel == 170]]

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

    values = [w_user, q2_user, cos_user,
              e_beam_user, eps_user,
              phi_user,
              x_axis_min, x_axis_max, interp_step_user, x_axis_label]
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
                            data['sigma_t'].tolist(), data['d_sigma_t'].tolist(), data['sigma_l'].tolist(),
                            data['d_sigma_l'].tolist(), data['sigma_tt'].tolist(), data['d_sigma_tt'].tolist(),
                            data['sigma_lt'].tolist(), data['d_sigma_lt'].tolist()])
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
                            data['sigma_t'].tolist(), data['d_sigma_t'].tolist(), data['sigma_l'].tolist(),
                            data['d_sigma_l'].tolist(), data['sigma_tt'].tolist(), data['d_sigma_tt'].tolist(),
                            data['sigma_lt'].tolist(), data['d_sigma_lt'].tolist()])
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
                            data['sigma_t'].tolist(), data['d_sigma_t'].tolist(), data['sigma_l'].tolist(),
                            data['d_sigma_l'].tolist(), data['sigma_tt'].tolist(), data['d_sigma_tt'].tolist(),
                            data['sigma_lt'].tolist(), data['d_sigma_lt'].tolist()])
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
                            data['sigma_t'].tolist(), data['d_sigma_t'].tolist(), data['sigma_l'].tolist(),
                            data['d_sigma_l'].tolist(), data['sigma_tt'].tolist(), data['d_sigma_tt'].tolist(),
                            data['sigma_lt'].tolist(), data['d_sigma_lt'].tolist()])

    if interpolation_method != -1:
        if our_method[3] == 1:
            calc_u_method = 0
        elif our_method[4] == 1:
            calc_u_method = 1
    if calc_u_method != -1:
        if (interpolation_method == 0) and (our_method[5] != 1):
            calc_cross_section_method = 1
        elif (interpolation_method != 0) and (our_method[5] == 1):
            calc_cross_section_method = 2


    res_x_axis_values = []
    res_sigma_TT, res_sigma_LT, res_sigma_T, res_sigma_L = [], [], [], []
    res_dsigma_TT, res_dsigma_LT, res_dsigma_T, res_dsigma_L, = [], [], [], []

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
                           'sigma_tt': res_sigma_TT,
                           'sigma_lt': res_sigma_LT,
                           'sigma_t': res_sigma_T,
                           'sigma_l': res_sigma_L,
                           'd_sigma_tt': res_dsigma_TT,
                           'd_sigma_lt': res_dsigma_LT,
                           'd_sigma_t': res_dsigma_T,
                           'd_sigma_l': res_dsigma_L})

    if interpolation_method == 0:
        if len(res_df) > 0:
            len_val_5 = len(values[5])
            res_df = pd.DataFrame({'x_axis_values': values[5],
                                   'sigma_tt': res_sigma_TT * len_val_5,
                                   'sigma_lt': res_sigma_LT * len_val_5,
                                   'sigma_t': res_sigma_T * len_val_5,
                                   'sigma_l': res_sigma_L * len_val_5,
                                   'd_sigma_tt': res_dsigma_TT * len_val_5,
                                   'd_sigma_lt': res_dsigma_LT * len_val_5,
                                   'd_sigma_t': res_dsigma_T * len_val_5,
                                   'd_sigma_l': res_dsigma_L * len_val_5})

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
            nu = (tmp_w ** 2 + tmp_q2 - mp * mp) / (2 * mp)
            res_df['eps'] = 1 / (1 + 2 * (nu ** 2 + tmp_q2) / (4 * (tmp_ebeam - nu) * tmp_ebeam - tmp_q2))

        if calc_u_method != -1:
            res_df['res_A'] = res_df['sigma_t'] + res_df['eps'] * res_df['sigma_l']
            res_df['d_res_A'] = ((res_df['d_sigma_t'] ** 2) + ((res_df['eps'] * res_df['d_sigma_l']) ** 2)) ** 0.5
            res_df['res_B'] = res_df['eps'] * res_df['sigma_tt']
            res_df['d_res_B'] = res_df['eps'] * res_df['d_sigma_tt']
            res_df['res_C'] = ((2 * res_df['eps'] * (res_df['eps'] + 1)) ** 0.5) * res_df['sigma_lt']
            res_df['d_res_C'] = ((2 * res_df['eps'] * (res_df['eps'] + 1)) ** 0.5) * res_df['d_sigma_lt']

        if calc_cross_section_method == 2:
            phi = values[5]
            res_df['res_cross_sect'] = res_df['res_A'] + res_df['res_B'] * np.cos(2 * phi) + \
                                       res_df['res_C'] * np.cos(phi)
            res_df['d_res_cross_sect'] = (res_df['d_res_A'] ** 2 + (res_df['d_res_B'] * np.cos(2 * phi)) ** 2 +
                                          (res_df['d_res_C'] * np.cos(phi)) ** 2) ** 0.5
        elif calc_cross_section_method == 1:
            phi = res_df['x_axis_values'].copy() * (np.pi / 180)
            res_df['res_cross_sect'] = res_df['res_A'] + res_df['res_B'] * np.cos(2 * phi) + \
                                       res_df['res_C'] * np.cos(phi)
            res_df['d_res_cross_sect'] = (res_df['d_res_A'] ** 2 + (res_df['d_res_B'] * np.cos(2 * phi)) ** 2 +
                                          (res_df['d_res_C'] * np.cos(phi)) ** 2) ** 0.5

    res_df.sort_values(by='x_axis_values', inplace=True)
    x_axis_label = values[-1]

    return res_df, interpolation_method, calc_u_method, calc_cross_section_method, x_axis_label

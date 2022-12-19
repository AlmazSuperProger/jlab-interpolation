import pandas as pd


def get_exp_data(exp_data_df, particle_class, interp_method_user, w_user, q2_user, cos_user, x_min_user, x_max_user):
    if particle_class == 'Pin':
        exp_data_df = exp_data_df[(exp_data_df.Channel == 8) | (exp_data_df.Channel == 14) |
                                  (exp_data_df.Channel == 41) | (exp_data_df.Channel == 141)].copy()
    elif particle_class == 'Pi0P':
        exp_data_df = exp_data_df[
            (exp_data_df.Channel == 9) | (exp_data_df.Channel == 37) | (exp_data_df.Channel == 170)].copy()


    if interp_method_user == 1:  # W Q2 filled
        exp_data_df = exp_data_df[
            (exp_data_df['w_average'] == float(w_user)) & (exp_data_df['q2_average'] == float(q2_user))].copy()
        exp_data_df['x_axis_values'] = exp_data_df['Cos(theta)']
    elif interp_method_user == 2:  # W cos filled
        exp_data_df = exp_data_df[
            (exp_data_df['w_average'] == float(w_user)) & (exp_data_df['Cos(theta)'] == float(cos_user))].copy()
        exp_data_df['x_axis_values'] = exp_data_df['q2_average']
    elif interp_method_user == 3:  # W cos filled
        exp_data_df = exp_data_df[
            (exp_data_df['q2_average'] == float(q2_user)) & (exp_data_df['Cos(theta)'] == float(cos_user))].copy()
        exp_data_df['x_axis_values'] = exp_data_df['w_average']

    else:
        exp_data_df = pd.DataFrame(
            columns=['x_axis_values', 'w_average', 'q2_average', 'Cos(theta)', 'sigma_t', 'd_sigma_t',
                     'sigma_l', 'd_sigma_l', 'sigma_tt', 'd_sigma_tt', 'sigma_lt', 'd_sigma_lt', 'eps'])


    # try:
    #     exp_data_df['x_axis_values'] = exp_data_df[(exp_data_df['x_axis_values'] <= float(x_max_user)) &
    #                                                (exp_data_df['x_axis_values'] >= float(x_min_user))]
    #     exp_data_df=exp_data_df.sort_values(by='exp_data_df')
    # except:
    #     pass


    return exp_data_df

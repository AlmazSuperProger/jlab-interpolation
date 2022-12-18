#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import unicode_literals

import cgi
import pandas as pd
from interpolation.interpolate import make_interpolation
from interpolation.get_exp_data import get_exp_data
from make_html.make_html import print_end, print_head, create_form_template
from make_html.create_graph_html import create_graph_html

gettext = cgi.FieldStorage()
particle_class_ext = gettext.getfirst("particle", "empty")
w_user_ext = gettext.getfirst("w", "empty")
q2_user_ext = gettext.getfirst("q2", "empty")
cos_user_ext = gettext.getfirst("cos", "empty")
e_beam_user_ext = gettext.getfirst("eBeam", "empty")
eps_user_ext = gettext.getfirst("eps", "empty")
phi_user_ext = gettext.getfirst("phi", "empty")
interp_step_user_ext = float(gettext.getfirst("grid_step_user", "0.01"))
x_axis_min_ext = gettext.getfirst("x_min", "empty")
x_axis_max_ext = gettext.getfirst("x_max", "empty")

# particle_class_ext = "Pin"
# w_user_ext = "1.23"
# q2_user_ext = "0.5"
# cos_user_ext = "empty"
# e_beam_user_ext = "empty"
# eps_user_ext = "0.92"
# phi_user_ext = "5.75"
# interp_step_user_ext = float("0.1")
# x_axis_min_ext = "empt"
# x_axis_max_ext = "val"

df_ext = pd.read_csv('final_table.csv', header=None, sep='\t',
                     names=['Channel', 'MID', 'Wmin', 'Wmax', 'Q2min', 'Q2max', 'Cos(theta)', 'sigma_t', 'd_sigma_t',
                            'sigma_l', 'd_sigma_l', 'sigma_tt', 'd_sigma_tt', 'sigma_lt', 'd_sigma_lt', 'eps'])
df_ext['w_average'] = (df_ext['Wmin'] + df_ext['Wmax']) / 2
df_ext['q2_average'] = (df_ext['Q2min'] + df_ext['Q2max']) / 2

values_to_func = [particle_class_ext, w_user_ext, q2_user_ext, cos_user_ext, e_beam_user_ext, eps_user_ext,
                  phi_user_ext, interp_step_user_ext, x_axis_min_ext, x_axis_max_ext]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    res_df, interp_method, calc_u_method, calc_cr_sect_method, x_axis_label = make_interpolation(df_ext,
                                                                                                 particle_class_ext,
                                                                                                 w_user_ext,
                                                                                                 q2_user_ext,
                                                                                                 cos_user_ext,
                                                                                                 e_beam_user_ext,
                                                                                                 eps_user_ext,
                                                                                                 phi_user_ext,
                                                                                                 interp_step_user_ext,
                                                                                                 x_axis_min_ext,
                                                                                                 x_axis_max_ext)

    experimental_data_df = get_exp_data(df_ext, particle_class_ext, interp_method,
                                        w_user_ext, q2_user_ext, cos_user_ext,
                                        x_axis_min_ext, x_axis_max_ext)

    print_head()
    create_form_template(values_to_func)
    create_graph_html(res_df,experimental_data_df, interp_method, calc_u_method, calc_cr_sect_method, x_axis_label)
    print_end()

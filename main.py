#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import unicode_literals

import pandas as pd
from interpolation.interpolate import make_interpolation
from make_html.make_html import print_end, print_head
from make_html.create_graph_html import create_graph_html

# gettext = cgi.FieldStorage()
# particle_class_ext = gettext.getfirst("particle", "empty")
# w_user_ext = gettext.getfirst("w", "empty")
# q2_user_ext = gettext.getfirst("q2", "empty")
# cos_user_ext = gettext.getfirst("cos", "empty")
# e_beam_user_ext = gettext.getfirst("eBeam", "empty")
# eps_user_ext = gettext.getfirst("eps", "empty")
# phi_user_ext = gettext.getfirst("phi", "empty")
# interp_step_user_ext = gettext.getfirst("grid_step_user", "empty")
# x_axis_min_ext = gettext.getfirst("this_min_value", "empty")
# x_axis_max_ext = gettext.getfirst("this_max_value", "empty")
# x_axis_label_ext = "empty"

particle_class_ext = "Pin"
w_user_ext = "1.3"
q2_user_ext = "0.5"
cos_user_ext = "empty"
e_beam_user_ext = "empty"
eps_user_ext = "0.92"
phi_user_ext = "5.75"
interp_step_user_ext = float("0.1")
x_axis_min_ext = "empt"
x_axis_max_ext = "val"
x_axis_label_ext = "empty"

df_ext = pd.read_csv('final_table.csv', header=None, sep='\t',
                     names=['Channel', 'MID', 'Wmin', 'Wmax', 'Q2min', 'Q2max', 'Cos(theta)', 'Sigma_T', 'dSigma_T',
                            'Sigma_L', 'dSigma_L', 'Sigma_TT', 'dSigma_TT', 'Sigma_LT', 'dSigma_LT', 'eps'])
df_ext = df_ext.assign(w_average=((df_ext['Wmin'] + df_ext['Wmax']) / 2))
df_ext = df_ext.assign(q2_average=((df_ext['Q2min'] + df_ext['Q2max']) / 2))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    res_df, interpolation_method, calc_u_method, calc_cross_section_method, x_axis_label = make_interpolation(df_ext,
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

    print_head()
    create_graph_html(res_df, interpolation_method, calc_u_method, calc_cross_section_method, x_axis_label)
    print_end()

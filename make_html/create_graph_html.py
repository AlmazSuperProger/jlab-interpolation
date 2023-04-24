from make_plot.make_plot import graph_inerp_exp_maid
from interpolation.calculate_intgral import calculate_integral
from maid_almaz_version import calculate_maid_structure_func

import numpy as np

def create_graph_html(particle_ext,
                      res_df_ext,
                      exp_data_df,
                      interpolation_method_ext,
                      calc_u_method_ext,
                      calc_cross_section_method_ext,
                      x_axis_label_ext):
    # print(interpolation_method_ext)
    # print(calc_u_method_ext)
    # print(calc_cross_section_method_ext)

    maid_t, maid_l, maid_tt, maid_tl, d_maid_t, d_maid_l, d_maid_tt, d_maid_tl = [], [], [], [], [], [], [], []
    if interpolation_method_ext != -1:
        if len(res_df_ext) > 0:
            maid_t, maid_l, maid_tt, maid_tl, d_maid_t, d_maid_l, d_maid_tt, d_maid_tl = calculate_maid_structure_func(particle_ext, res_df_ext)


    if interpolation_method_ext != -1:
        if calc_cross_section_method_ext != -1:
            fig_cross_section = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'],
                                                     y_array=res_df_ext['res_cross_sect'],
                                                     d_y_array=res_df_ext['d_res_cross_sect'],
                                                     maid_y_array=[],
                                                     x_exp_data=[],
                                                     y_exp_data=[],
                                                     dy_exp_data=[],
                                                     layout_title='Differential Cross Section (mcbn/sterad)',
                                                     x_label=x_axis_label_ext)
            print("{}".format(fig_cross_section.to_html(full_html=False)))
            print("<br>")

        fig_sigma_tt = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'], y_array=res_df_ext['sigma_tt'],
                                            d_y_array=res_df_ext['d_sigma_tt'],
                                            maid_y_array=maid_tt,
                                            x_exp_data=exp_data_df['x_axis_values'],
                                            y_exp_data=exp_data_df['sigma_tt'],
                                            dy_exp_data=exp_data_df['d_sigma_tt'],
                                            layout_title='Transverse-Transverse structure function (mcbn/sterad)',
                                            x_label=x_axis_label_ext)

        fig_sigma_lt = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'], y_array=res_df_ext['sigma_lt'],
                                            d_y_array=res_df_ext['d_sigma_lt'],
                                            maid_y_array=maid_tl,
                                            x_exp_data=exp_data_df['x_axis_values'],
                                            y_exp_data=exp_data_df['sigma_lt'],
                                            dy_exp_data=exp_data_df['d_sigma_lt'],
                                            layout_title='Longitudinal-Transverse structure function (mcbn/sterad)',
                                            x_label=x_axis_label_ext)

        fig_sigma_t = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'], y_array=res_df_ext['sigma_t'],
                                           d_y_array=res_df_ext['d_sigma_t'],
                                           maid_y_array=maid_t,
                                           x_exp_data=exp_data_df['x_axis_values'],
                                           y_exp_data=exp_data_df['sigma_t'],
                                           dy_exp_data=exp_data_df['d_sigma_t'],
                                           layout_title='Transverse structure function (mcbn/sterad)',
                                           x_label=x_axis_label_ext)

        fig_sigma_l = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'], y_array=res_df_ext['sigma_l'],
                                           d_y_array=res_df_ext['d_sigma_l'],
                                           maid_y_array=maid_l,
                                           x_exp_data=exp_data_df['x_axis_values'],
                                           y_exp_data=exp_data_df['sigma_l'],
                                           dy_exp_data=exp_data_df['d_sigma_l'],
                                           layout_title='Longitudinal structure function (mcbn/sterad)',
                                           x_label=x_axis_label_ext)

        if interpolation_method_ext > 0:
            integral_values = calculate_integral(res_df_ext)
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
                    </tr></table>""".format(round(integral_values[0], 4),
                                            round(integral_values[4], 4),
                                            round(integral_values[1], 4),
                                            round(integral_values[5], 4),
                                            round(integral_values[2], 4),
                                            round(integral_values[6], 4),
                                            round(integral_values[3], 4),
                                            round(integral_values[7], 4)))

        print("<br><br>")
        print("{}".format(fig_sigma_tt.to_html(full_html=False)))
        print("<br>")
        print("{}".format(fig_sigma_lt.to_html(full_html=False)))
        print("<br>")
        print("{}".format(fig_sigma_t.to_html(full_html=False)))
        print("<br>")
        print("{}".format(fig_sigma_l.to_html(full_html=False)))


    else:
        pass

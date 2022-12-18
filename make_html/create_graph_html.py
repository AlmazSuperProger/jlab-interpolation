from make_plot.make_plot import graph_inerp_exp_maid


def create_graph_html(res_df_ext,
                      interpolation_method_ext,
                      calc_u_method_ext,
                      calc_cross_section_method_ext,
                      x_axis_label_ext):
    # print(interpolation_method_ext)
    # print(calc_u_method_ext)
    # print(calc_cross_section_method_ext)
    if interpolation_method_ext != -1:
        if calc_cross_section_method_ext != -1:
            fig_cross_section = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'],
                                                     y_array=res_df_ext['res_cross_sect'],
                                                     d_y_array=res_df_ext['d_res_cross_sect'],
                                                     maid_y_array=[],
                                                     x_exp_data=[], y_exp_data=[], dy_exp_data=[],
                                                     layout_title='cross_section',
                                                     x_label=x_axis_label_ext)
            print("{}".format(fig_cross_section.to_html(full_html=False)))
            print("<br>")

        fig_sigma_tt = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'], y_array=res_df_ext['sigma_TT'],
                                            d_y_array=res_df_ext['dsigma_TT'],
                                            maid_y_array=[],
                                            x_exp_data=[], y_exp_data=[], dy_exp_data=[], layout_title='sigma_TT',
                                            x_label=x_axis_label_ext)

        fig_sigma_lt = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'], y_array=res_df_ext['sigma_LT'],
                                            d_y_array=res_df_ext['dsigma_LT'],
                                            maid_y_array=[],
                                            x_exp_data=[], y_exp_data=[], dy_exp_data=[], layout_title='sigma_LT',
                                            x_label=x_axis_label_ext)

        fig_sigma_t = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'], y_array=res_df_ext['sigma_T'],
                                           d_y_array=res_df_ext['dsigma_T'],
                                           maid_y_array=[],
                                           x_exp_data=[], y_exp_data=[], dy_exp_data=[], layout_title='sigma_T',
                                           x_label=x_axis_label_ext)

        fig_sigma_l = graph_inerp_exp_maid(x_array=res_df_ext['x_axis_values'], y_array=res_df_ext['sigma_L'],
                                           d_y_array=res_df_ext['dsigma_L'],
                                           maid_y_array=[],
                                           x_exp_data=[], y_exp_data=[], dy_exp_data=[], layout_title='sigma_L',
                                           x_label=x_axis_label_ext)

        print("{}".format(fig_sigma_tt.to_html(full_html=False)))
        print("<br>")
        print("{}".format(fig_sigma_lt.to_html(full_html=False)))
        print("<br>")
        print("{}".format(fig_sigma_t.to_html(full_html=False)))
        print("<br>")
        print("{}".format(fig_sigma_l.to_html(full_html=False)))

    else:
        pass

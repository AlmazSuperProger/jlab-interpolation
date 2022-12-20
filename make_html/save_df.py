filename_construct = '/home/almaz/public_html/save_folder/structure_functions/'
get_file_name = 'https://clas.sinp.msu.ru/~almaz/save_folder/structure_functions/'


def save_data(df_to_save, particle, w_user_ext, q2_user_ext, cos_user_ext):
    try:
        filename = str(particle) + "_" + str(w_user_ext) + "_" + str(q2_user_ext) + "_" + str(cos_user_ext) + '.csv'
        filename = filename_construct + filename
        df_to_save.to_csv(filename, index=False, header=True)
    except:
        pass

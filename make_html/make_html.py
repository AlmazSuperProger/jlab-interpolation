#!/usr/bin/env python3
# -*- coding: utf-8 -*
from jinja2 import Template


def print_head():
    print("Content-type: text/html\n")
    print("""
    <!DOCTYPE HTML>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style type="text/css">
            A {
                text-decoration: none;
                color: red;
            }
    
            * {
                margin: 0;
            }
    
            .textBox {
                width: 1440px;
                height: 80px;
                margin: auto;
            }
    
            .imagesBox {
                width: 1440px;
                height: 900px;
                margin: auto;
            }
    
            .textBox2 {
                width: 1440px;
                height: 50px;
                margin: auto;
            }
    
            .tableBox1 {
                margin: auto;
                width: 1440px;
                height: 350px;
            }
    
            .checkbox_msg {
                color: blue;
            }
    
            td {
                text-align: center;
            }
    
            .first_box {
                background-color: rgba(200, 200, 200, 0.6);
                width: 1070px;
                height: 570px;
                margin: auto;
                border-radius: 10px;
                margin-bottom: 30px;
            }
    
    
            .box_in_the_box {
                background-color: rgba(100, 100, 100, 0.6);
                position: absolute;
                width: 300px;
                height: 520px;
                margin-top: 25px;
                margin-left: 25px;
                border-radius: 4px;
            }
    
            .second_box_in_the_box {
                position: absolute;
                border-radius: 4px;
                margin-left: 350px;
                margin-top: 25px;
                width: 700px;
                height: 520px;
                background-color: rgba(100, 100, 100, 0.6);
            }
    
            .left_box {
                position: absolute;
                border-radius: 4px;
                margin-left: 20px;
                margin-top: 10px;
                width: 330px;
                height: 430px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.6);
    
            }
    
    
            .right_box {
                position: absolute;
                border-radius: 4px;
                margin-left: 400px;
                margin-top: 10px;
                width: 280px;
                height: 450px;
                background-color: rgba(255, 255, 255, 0.6);
    
            }
    
            .input_small {
                width: 60px;
            }
    
    
            .left_sub_box {
    
                position: absolute;
                width: 130px;
                height: 280px;
    
    
            }
    
            .right_sub_box {
                margin-top: -40px;
                position: absolute;
                width: 130px;
                margin-left: 130px;
                height: 280px;
            }
    
            .select_hidden {
                visibility: hidden;
            }
    
    
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
    """)


def print_end():
    print("""
    </center>
    </body>
    </html>""")


def create_form_template(tmp_values):
    particle_class_ext = tmp_values[0]
    w_user_ext = tmp_values[1]
    q2_user_ext = tmp_values[2]
    cos_user_ext = tmp_values[3]
    e_beam_user_ext = tmp_values[4]
    eps_user_ext = tmp_values[5]
    phi_user_ext = tmp_values[6]
    interp_step_user_ext = tmp_values[7]
    x_axis_min_ext = tmp_values[8]
    x_axis_max_ext = tmp_values[9]

    this_file_name = str(particle_class_ext) + "_" + str(w_user_ext) + "_" + str(q2_user_ext) + "_" + str(
        cos_user_ext) + '.csv'
    file_dir_name = 'https://clas.sinp.msu.ru/~almaz/save_folder/structure_functions/'+this_file_name

    link_text="""<br> <a href="{}" download="{}">Download Your
            CSV file here</a> <br>""".format(file_dir_name, this_file_name)

    tmp_dict = {'val0': particle_class_ext,
                'val1': w_user_ext,
                'val2': q2_user_ext,
                'val3': cos_user_ext,
                'val4': e_beam_user_ext,
                'val5': eps_user_ext,
                'val6': phi_user_ext,
                'val7': interp_step_user_ext,
                'val8': x_axis_min_ext,
                'val9': x_axis_max_ext,
                'val10':link_text}

    tmp = Template("""
    <br><br><br><br>
    <form method="GET" action="https://clas.sinp.msu.ru/cgi-bin/almaz/jlab-interpolation/main.py">
        <div class="first_box">
            <div class="box_in_the_box">
                <br>
                <br>



            </div>
            <div class="second_box_in_the_box">
                <br>
                Interpolation features
                <div class="left_box">
                    <br>
                    <p>Введите W+Q2 или W+cos(theta) или Q2+cos(theta) и получите
                        зависимость структурных функций от третьей переменной (cos(theta)/W/Q2)</p>
                    <br>
                    (Введите дополнительно Ebeam/eps и phi и дополнительно получите зависимость сечения от третьей
                    переменной (cos(theta)/W/Q2))
                    <br>
                    ___________________________
                    <br>
                    <br>
                    Введите W+Q2+cos(theta), Ebeam/eps
                    и получите зависимость дифф сечения от угла phi
                    <br>
                    ___________________________
                    <br>
                    <br>
                    x_min и x_max позволяют вывести зависимости в необходимом диапазоне (обрезать график по оси x)
                    <br>
                    <br>
                    
                    {{ val10 }}

                </div>
                <div class="right_box">
                    <br>
                    <p>W (GeV)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <input type="text" name="w"
                                                                                placeholder="W value"
                                                                                value="{{ val1 }}"></p>
                    <br>
                    <p>Q2 (GeV2)&nbsp;&nbsp;&nbsp;&nbsp; <input type="text" name="q2" placeholder="Q2 value"
                                                                value="{{ val2 }}"></p>
                    <br>
                    <p>cos(theta) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<input type="text" name="cos"
                                                                                   placeholder="Cos theta value"
                                                                                   value="{{ val3 }}">
                    </p>
                    <br>
                    <p>phi (degree) &nbsp;&nbsp;&nbsp;<input type="text" name="phi" placeholder="phi value"
                                                             value="{{ val6 }}"></p>
                    <br>
                    Enter BeamEnergy OR eps (optional)
                    <br>
                    <br>
                    <p>
                        Energy:
                        <input class="input_small" type="text" name="eBeam" placeholder="MeV" value="{{ val4 }}">
                        &nbsp;&nbsp;&nbsp;&nbsp;eps:
                        <input class="input_small" type="text" name="eps" placeholder="eps" value="{{ val5 }}">
                    </p>
                    <br>
                    x_axis min and max values (optional)
                    <br>
                    <br>
                    <p>x_min <input class="input_small" type="text" name="x_min" placeholder="x_min" value="{{ val8 }}">
                        &nbsp;&nbsp;&nbsp;x_min
                        <input class="input_small" type="text" name="x_max" placeholder="x_max" value="{{ val9 }}">
                    </p>
                    <br>
                    <p> interpolation step &nbsp;&nbsp;&nbsp;&nbsp;
                        <input class="input_small" type="text" name="grid_step_user" placeholder="grid step"
                               value="{{ val7 }}">
                    </p>
                    <br>
                    <p> reaction channel:
                        <select class="select" name="particle" size="1">
                            <option value="Pin">gvp---&gt;π⁺n</option>
                            <option value="Pi0P">gvp---&gt;π⁰p</option>
                        </select>
                    </p>
                    <br>
                    <p><input class="button" class="submitbutton" type="submit" value="Run"></p>
                </div>
            </div>
            nasrtdinov.ag17@physics.msu.ru
        </div>

    </form>
    
    <br>
    
    
    <br><br>""")

    tmp_html = tmp.render(tmp_dict)
    print(tmp_html)

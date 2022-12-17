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

    tmp_dict = {'val0': particle_class_ext,
                'val1': w_user_ext,
                'val2': q2_user_ext,
                'val3': cos_user_ext,
                'val4': e_beam_user_ext,
                'val5': eps_user_ext,
                'val6': phi_user_ext,
                'val7': interp_step_user_ext,
                'val8': x_axis_min_ext,
                'val9': x_axis_max_ext}

    tmp = Template("""
    <form method="GET" action="https://clas.sinp.msu.ru/cgi-bin/almaz/jlab-interpolation/main.py">
        <p><input type="text" name="eBeam" placeholder="Ebeam(GeV)" value="{{ val4 }}">
            <input type="text" name="eps" placeholder="eps" value="{{ val5 }}"></p>
        <br>
        <input type="text" name="w" placeholder="W(GeV)" value="{{ val1 }}">
        <input type="text" name="q2" placeholder="Q2(GeV2)" value="{{ val2 }}">
        <input type="text" name="cos" placeholder="Cos(theta)" value="{{ val3 }}">
        <input type="text" name="phi" placeholder="phi(degree)" value="{{ val6 }}">
        <input type="text" name="grid_step_user" placeholder="grid step" value="{{ val7 }}">
        <br>
        <input type="text" name="x_min" placeholder="x_axis_min" value="{{ val8 }}">
        <input type="text" name="x_max" placeholder="x_axis_max" value="{{ val9 }}">
        <br>
        <select class="select" name="particle" size="1">
            <option value="Pin">gvp---&gt;π⁺n</option>
            <option value="Pi0P">gvp---&gt;π⁰p</option>
        </select>
        <br>
        <p><input class="button" class="submitbutton" type="submit" value="Run"></p>
        <br>
    </form>""")

    tmp_html = tmp.render(tmp_dict)
    print(tmp_html)

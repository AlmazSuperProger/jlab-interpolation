#!/usr/bin/env python3
# -*- coding: utf-8 -*

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
    HELLO
    """)



def print_end():
    print("""
    </center>
    </body>
    </html>""")

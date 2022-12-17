import plotly.graph_objs as go

def graph_inerp_exp_maid(x_array=[], y_array=[], d_y_array=[],
                         maid_y_array=[],
                         x_exp_data=[], y_exp_data=[], dy_exp_data=[],
                         layout_title='', x_label=''):
    trace_interp = go.Scatter(
        x=x_array,
        y=y_array,
        error_y=dict(
            type='data',
            array=d_y_array,
            color='rgba(100, 100, 255, 0.6)',
            thickness=1.5,
            width=3),
        name='Interpolation',
        marker_size=1)

    trace_maid = go.Scatter(
        x=x_array,
        y=maid_y_array,
        name='maid 2007',
        marker_size=3)

    trace_exp = go.Scatter(
        mode='markers',
        x=x_exp_data,
        y=y_exp_data,
        name='Experiment',
        marker=dict(color='rgba(100, 100, 100, 1)',
                    symbol='square'),
        error_y=dict(
            type='data',
            array=dy_exp_data,
            color='rgba(100, 100, 100, 1)',
            thickness=1.5,
            width=3),

        marker_size=10)

    data = [trace_interp, trace_maid, trace_exp]

    fig = go.Figure(data=data)
    fig.layout.height = 700
    fig.layout.width = 1000
    fig.layout.title = layout_title

    fig.layout.yaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.5,
        title=layout_title,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))
    fig.layout.xaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.2,
        title=x_label,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))

    return fig
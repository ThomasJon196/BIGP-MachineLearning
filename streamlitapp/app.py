import streamlit as st
import pandas as pd
import numpy as np
# import plotly.express as px
import plotly.graph_objs as go

from gp_utils import gp_regression

# Streamlit setup
st.set_page_config(layout="wide")


# Helper functions
def create_initial_plot():

    # nx, ny = (5, 5)
    # x = np.linspace(-5, 5, nx)
    # y = np.linspace(-5, 5, ny)
    # xv, yv = np.meshgrid(x, y)
    # x=xv.flatten()
    # y=yv.flatten()
    x = [-3, -1, 1, 3]
    y = [-2, 0, -3, 5]


    # Can write inside of things using with!
    # with st.expander('Plot'):

    # Select other Plotly events by specifying kwargs
    # fig = px.line(x=[1], y=[1])
    # selected_points = plotly_events(fig, click_event=False, hover_event=True)

    return fig


def retrieve_user_input():
    points = st.session_state.input_points

    if len(points) != 0:
        x = []
        y = []

        for point in points:
            x.append(point[0])
            y.append(point[1])
        return x, y
    else:
        return [], []


def add_gp_regression_to_trace(fig):
    
    X_pred = np.linspace(-5, 5, 200)

    x,y = retrieve_user_input()

    if len(x) != 0:

        observations = {}
        observations["x"] = np.asarray(x)
        observations["Y"] = np.asarray(y)

        mean, conf_interval = gp_regression(observations, X_pred)

        fig.add_trace(go.Scatter(x=X_pred, y=mean, mode='lines'))
        


# Main script
st.title('Gaussian Processes Regression')

if 'input_points' not in st.session_state:
    st.session_state.input_points = []  


if st.button('Sample random point'):

    x = np.random.rand() * 20 - 10
    y = np.random.rand() * 10 - 5

    st.session_state.input_points.append([x, y])

    st.write('Sampled:', x, y)

n_predictions = 100

x_pred = np.linspace(-10, 10, n_predictions)
y = np.linspace(-5, 5, 100)

# fig = px.line(x=x, y=y)

## Plotting
fig = go.Figure()

fig.update_yaxes(range=[-5, 5])
fig.update_xaxes(range=[-10, 10])
fig.update_layout(
autosize=False,
width=1200,
height=500,)


## Retrieve observations
x, y = retrieve_user_input()

# GP Calc

def calc_gp_regression(x, y, x_pred):
    observations = {}
    observations["x"] = np.asarray(x)
    observations["Y"] = np.asarray(y)

    mean, conf_interval = gp_regression(observations, x_pred)
    return mean, conf_interval

mean, conf_interval = calc_gp_regression(x, y, x_pred)


# GP plot

# def plot_gp():
fig.add_trace(go.Scatter(x=x_pred, y=mean, mode='lines'))

fig.add_trace(go.Scatter(x=x, y=y, mode='markers',marker=dict(size=10)))



st.plotly_chart(fig)


# Debug info






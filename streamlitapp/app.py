import streamlit as st
import pandas as pd
import numpy as np
from streamlit_plotly_events import plotly_events
import plotly.graph_objs as go
import plotly.express as px
from gp_utils import gp_regression
from streamlit_autorefresh import st_autorefresh


st.set_page_config(layout="wide")




# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
# count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

st.title('Gaussian Processes Regression')

if 'input_points' not in st.session_state:
    st.session_state.input_points = []  


# Helper functions
def create_meshgrid():

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
    fig = go.Figure()

    fig.update_layout(
    autosize=False,
    width=1200,
    height=500,)

    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',marker=dict(size=20, color="black")))

    # Select other Plotly events by specifying kwargs
    # fig = px.line(x=[1], y=[1])
    # selected_points = plotly_events(fig, click_event=False, hover_event=True)

    return fig


def update_user_input(current_input):
    if len(current_input) != 0:
        
        x = current_input[0].get("x")
        y = current_input[0].get("y")
        point = [x, y]
        if point in st.session_state.input_points:
            st.session_state.input_points.remove(point)
        else:
            st.session_state.input_points.append(point)
        
        st.experimental_rerun()


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
st.write("Select the black dots to start the regression.")

fig = create_meshgrid()

add_gp_regression_to_trace(fig)

selected_point = plotly_events(fig)

update_user_input(selected_point)




# Debug info
st.write("Saved points:", st.session_state.input_points)

st.write(selected_point)



# st.experimental_rerun()

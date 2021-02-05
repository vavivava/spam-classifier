import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from pandas_datareader import data as web
from datetime import datetime as dt
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame()

fig = px.bar(df, x="col1", y="col2", color="col3", barmode="group")

app.layout = html.Div(children=[
    html.H3(children='EMAIL SPAM - CHECK YOUR MESSAGE!',
            style={'textAlign': 'center'}),

    html.Div(children='''
        Machine learning model for email spam filtering. Choose the model and type the message to check 
        if this is a spam or not: 
    ''', style={'textAlign': 'center'}),

    html.Label('Choose'),
    dcc.RadioItems(
        options=[
            {'label': 'Word2Vec [to get similar words]', 'value': 'word'},
            {'label': 'Logistic Model [bow]', 'value': 'bow'},
            {'label': 'Logistic Model [tfidf]', 'value': 'tfidf'}
        ],
        value='word',
        labelStyle={'display': 'inline-block'}
    ),

    html.Label('Text Input'),
    dcc.Input(value='', type='text'),
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)

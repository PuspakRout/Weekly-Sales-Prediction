import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import os
import joblib
import pymysql
from codes import *
import subprocess


# ------------------------ SQL Connection --------------------------#

db_user = os.environ.get('CLOUD_SQL_USERNAME')
db_password = os.environ.get('CLOUD_SQL_PASSWORD')
db_name = os.environ.get('CLOUD_SQL_DATABASE_NAME')
db_connection_name = os.environ.get('CLOUD_SQL_CONNECTION_NAME')

    # When deployed to App Engine, the `GAE_ENV` environment variable will be
    # set to `standard`
if os.environ.get('GAE_ENV') == 'standard':
    # If deployed, use the local socket interface for accessing Cloud SQL
    unix_socket = '/cloudsql/{}'.format(db_connection_name)
    cnx = pymysql.connect(user=db_user, password=db_password,
                            unix_socket=unix_socket, db=db_name)
else:
    # If running locally, use the TCP connections instead
    # Set up Cloud SQL Proxy (cloud.google.com/sql/docs/mysql/sql-proxy)
    # so that your application can use 127.0.0.1:3306 to connect to your
    # Cloud SQL instance
    host = '127.0.0.1'
    cnx = pymysql.connect(user='root', password='Pixel', host=host, db='Forecast')

# -------------------------- LOAD DATA ---------------------------- #

if os.path.isfile('plot_df.obj'):
    plot_df = joblib.load('plot_df.obj')

else:
    data = Data()
    data.read_from_sql_db(cnx)
    data.preprocessing()
    data.test_preprocessing()
    model = modelling(data)
    model.fit()
    model.predict()
    plot_df = model.data.create_plot_df()
    joblib.dump(model.data.plot_df,'plot_df.obj')


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


# -------------------------- DASH ---------------------------- #


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder='assets')
server = app.server

app.config.suppress_callback_exceptions = True


# ------------------------------- APP CALLBACK -----------------------------#


@app.callback(
    [Output('weekly-sales', 'figure'), Output('store-dept', 'children')],
    [Input('Store', 'value'), Input('Dept', 'value')])
def update_fig(Store, Dept):
    df = plot_df[(plot_df.Store==Store) & (plot_df.Dept==Dept)]
    df1 =df[df.Date<=(df.Date.max()-np.timedelta64(7,'D'))]
    df2 = df[df.Date>=(df.Date.max()-np.timedelta64(7,'D'))]
    return [{
            'data': [
                dict(x = df1.Date, y = df1.Weekly_Sales, type='line', name='Historic Sales'),
                dict(x = df2.Date, y = df2.Weekly_Sales, type='line', name='Predicted Sales')
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        },'Store : {}, Dept : {}'.format(Store, Dept)]


# -------------------------- PROJECT DASHBOARD ---------------------------- #


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Weekly Sales',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    
    html.Div(        
        children=[
            dcc.Input(id="Store", type="number", placeholder=""),
            dcc.Input(id="Dept", type="number", placeholder="", debounce=True), html.Div(id='output')],
        
        style=dict(display='flex', justifyContent='left')
        
    ),

    html.Div(id='store-dept', 
             style={
                    'textAlign': 'center',
                    'color': colors['text']
                    }),

    dcc.Graph(
        id='weekly-sales')
])


# -------------------------- MAIN ---------------------------- #

#subprocess.Popen('python3 scheduled.py')   # This is used to periodic update of weekly sales

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)

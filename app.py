import dash
import dash_design_kit as ddk
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import nrel_dash_components as ndc
import deckgl_ly as dgl

# Data/file handling imports
import pathlib
import glob
import pandas as pd
import numpy as np
import sys
import time
import datetime
import globalsUpdater as gu
import globals as gl
import math

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
# Initialization
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#

# Set the path to the data
DATA_PATH = pathlib.Path(__file__).parent.joinpath("./data").resolve()
SCENARIO_PATH = DATA_PATH.joinpath("50Scen_3days 2")


# Read in the network
bus = pd.read_csv(DATA_PATH.joinpath("RTS/bus.csv"))
bus = bus[['Bus ID','lat','lng']]
branch = pd.read_csv(DATA_PATH.joinpath("RTS/branch.csv"))
branch = branch[['UID','From Bus','To Bus']]
#gen = pd.read_csv(DATA_PATH.joinpath("RTS/gen.csv"))
gu.setBus(bus)
gu.setBranch(branch)
print("bus", bus, branch)




#------------------------------------------------#
# Create app and layout
app = dash.Dash(__name__, title='RTS', update_title='RTS')
server = app.server  # expose server variable for Dash server deployment
app.layout =  ndc.NRELApp(
     appName="RTS Grid Vis",
     description="",
     children=[
        dcc.Store(id='directory', storage_type='memory'),
        dcc.Store(id='busFile', storage_type='memory'),
        dcc.Store(id='busLossExtents', storage_type='memory'),
        dcc.Store(id='busThermalExtents', storage_type='memory'),
        dcc.Store(id='busRenewableExtents', storage_type='memory'),
        dcc.Store(id='summaryExtents', storage_type='memory'),
        dcc.Store(id='branchFile', storage_type='memory'),
        dcc.Store(id='timestep', storage_type='memory'),
        dcc.Store(id='busVariable', storage_type='memory'),
        dcc.Store(id='branchVariable', storage_type='memory'),
        dcc.Store(id='busDataColor', storage_type='memory'),
        dcc.Store(id='branchDataColor', storage_type='memory'),
        dcc.Interval(
            id='time-interval',
            interval=1*100, # in milliseconds
            n_intervals=0
        ),
        html.Hr(),
])



#------------------------------------------------#
# Callbacks to update the scenario directory
#------------------------------------------------#
# @app.callback(Output('directory', 'data'),
#                #Output('bus-variable-dropdown', 'options'),
#                #Output('branch-variable-dropdown', 'options'),
#                #Output('branch-variable-dropdown', 'disabled'),
#                #Output('branch-variable-dropdown', 'placeholder'),],
#               Input('scenario-dropdown', 'value'),
#               prevent_initial_call=True)
# def updateScenarioDirectory(value):
#     if(value == None):
#         raise PreventUpdate()
#

    # branchDisabled = False
    # branchPlaceholder = "Select..."
    #
    # # Get the list of data sets in this directory
    # dataFiles = glob.glob(str(SCENARIO_PATH.joinpath(value+"/*.csv")))
    #
    # dataFiles.sort()
    # res = list(map(lambda st: str.replace(st, str(SCENARIO_PATH.joinpath(value+"/")), ""), dataFiles))
    # res = list(map(lambda st: str.replace(st, "/", ""), res))
    #
    # # Find the bus or branch files that exist in the directory
    # busFiles = []
    # for i in res :
    #     if(i in busVariableFiles):
    #         busFiles.append(i)
    # branchFiles = []
    # for i in res :
    #     if(i in branchVariableFiles):
    #         branchFiles.append(i)
    #
    # busOptions=[{ "label": busVariableFiles[i]['title'], "value": i } for i in busFiles]
    # branchOptions= [{ "label": branchVariableFiles[i]['title'], "value": i } for i in branchFiles]
    # if(len(branchOptions) == 0):
    #     branchOptions =[{"label": "None", "value": 0}]
    #     branchDisabled = True
    #     branchPlaceholder = "None"
    #
    # return value#, busOptions, branchOptions, branchDisabled, branchPlaceholder





if __name__ == '__main__':
    app.run_server(debug=True)

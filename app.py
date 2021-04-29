import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
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
import defines as dn

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
# Initialization
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#

# ----------------------------------------------------------- #
# Read in the data and do some initial processing

# Set the path to the data
DATA_PATH = pathlib.Path(__file__).parent.joinpath("./data").resolve()
SCENARIO_PATH = DATA_PATH.joinpath("50Scen_3days 2")

# Read in the network
bus = pd.read_csv(DATA_PATH.joinpath("RTS/bus.csv"))
bus = bus[['Bus ID','lat','lng']]
branch = pd.read_csv(DATA_PATH.joinpath("RTS/branch.csv"))
branch = branch[['UID','From Bus','To Bus']]
#gen = pd.read_csv(DATA_PATH.joinpath("RTS/gen.csv"))

# Process the branch data
fromPos = []
toPos = []
for index, row in branch.iterrows():
    # Get the from bus and the to bus
    fromBus = bus.loc[bus['Bus ID'] == row['From Bus']]
    toBus = bus.loc[bus['Bus ID'] == row['To Bus']]
    fromPos.append([float(fromBus['lng']),float(fromBus['lat'])])
    toPos.append([float(toBus['lng']),float(toBus['lat'])])
branch['fromPos'] = fromPos
branch['toPos'] = toPos

# Get the list of summary data sets
summary_files = glob.glob(str(SCENARIO_PATH.joinpath("*.csv")))
summary_files.sort()
res = list(map(lambda st: str.replace(st, str(SCENARIO_PATH.joinpath("")), ""), summary_files))
res = list(map(lambda st: str.replace(st, "_", " "), res))
res = list(map(lambda st: str.replace(st, "/", ""), res))
listOfSummaryResults = list(map(lambda st: str.replace(st, ".csv", ""), res))

# Get the array of timeframes
summary =  pd.read_csv(SCENARIO_PATH.joinpath("ac_results.csv"))
timeSteps = summary['DateTime'].to_list()
daterange = timeSteps

# ----------------------------------------------------------- #
# Create the initial map state
viewState ={
    "longitude": -116,
    "latitude": 34.65,
    "zoom": 6,
    "maxZoom": 20,
    "pitch": 0,
    "bearing": 0
}
scatterLayer = {
    "type": "ScatterplotLayer",
    "id": 'scatterplot-layer2',
    "data": bus.to_dict(orient='records'),
    "pickable": False,
    "opacity": 0.15,
    "stroked": True,
    "filled": True,
    "radiusScale": 1,
    "radiusMinPixels": 5,
    "lineWidthMinPixels": 1,
    "getPosition": "function(d){return [d['lng'], d['lat']]}",
    "getRadius": 25,
    "getLineColor": [47, 79, 79],
    "getFillColor": [248, 248, 255]
}
lineLayer = {
    "type": "LineLayer",
    "id": 'line-layer',
    "data": branch.to_dict(orient='records'),
    "pickable": False,
    "opacity": 0.15,
    "widthScale":2,
    "widthMinPixels":1,
    "getWidth": 1,
    "getSourcePosition": "function(d){return d['fromPos']}",
    "getTargetPosition": "function(d){return d['toPos']}",
    "getColor": [47, 79, 79]
}
initialLayers = [lineLayer,scatterLayer]

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
# Layout
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#

#------------------------------------------------#
# Create app and layout
app = dash.Dash(__name__, title='RTS', update_title='', external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # expose server variable for Dash server deployment
app.layout =  ndc.NRELApp(
     appName="Multi-Fidelity Economic Dispatch Visualizations",
     description="",
     children=[
        dcc.Store(id='mode', storage_type='memory'),
        dcc.Store(id='runsFile', storage_type='memory'),
        dcc.Store(id='compareFile', storage_type='memory'),
        dcc.Store(id='basicsFile', storage_type='memory'),
        dcc.Store(id='timestep', storage_type='memory'),
        dcc.Store(id='summaryExtents', storage_type='memory'),
        dcc.Interval(
            id='time-interval',
            interval=1*100, # in milliseconds
            n_intervals=0
        ),
        html.Hr(),
        html.Div(className='columns', children=[
            html.Div(className='column is-one-quarter notification', children=[
                    dbc.Button("Investigate Power Flow Runs", id="scenario-fade-button", className="mb-3"),
                    dbc.Fade(
                        dbc.Card(
                            dbc.CardBody(children=[
                                html.H4("Select Run to Explore", className="card-title"),
                                #html.P("Choose a scenario, results ", className="card-text"),
                                # Left UI tiles
                                html.Div(className='tile is-parent is-vertical is-12', children=[
                                    #html.P(className="title is-size-6", children=["Power Flow Scenario:"]),
                                    html.Div(className="subtitle", children=[
                                        dcc.Dropdown(
                                            id="scenario-dropdown",
                                            style={"fontSize": "14px"},
                                            options=[{ "label": dn.scenarioDirs[i]["label"], "value": i} for i in dn.scenarioDirs.keys()],
                                        ),
                                    ]),
                                ]),
                            ])
                        ),
                        id="scenario-fade",
                        is_in=True,
                        appear=False,
                        className="mb-3",
                    ),
                    dbc.Button("Compare Power Flow Runs", id="comparison-fade-button", className="mb-3"),
                    dbc.Fade(
                        dbc.Card(
                            dbc.CardBody(children=[

                                html.H4("Choose runs for comparison.", className="card-text"),


                                html.P(className="title is-size-6", children=["Scenario 1:"]),
                                html.Div(className="subtitle", children=[
                                    dcc.Dropdown(
                                        id="scenario-1-dropdown",
                                        style={"fontSize": "14px"},
                                        options=[{ "label": dn.scenarioDirs[i]["label"], "value": i} for i in dn.scenarioDirs.keys()],
                                    ),
                                ]),
                                html.P(className="title is-size-6", children=["Scenario 2:"]),
                                html.Div(className="subtitle", children=[
                                    dcc.Dropdown(
                                        id="scenario-2-dropdown",
                                        style={"fontSize": "14px"},
                                        options=[{ "label": dn.scenarioDirs[i]["label"], "value": i} for i in dn.scenarioDirs.keys()],
                                    ),
                                ]),
                                html.P("Choose a map comparison operator and a variable."),
                                html.P(className="title is-size-6", children=["Operator:"]),
                                html.Div(className="subtitle", children=[
                                    dcc.Dropdown(
                                        id="operator-dropdown",
                                        style={"fontSize": "14px"},
                                        options=[{ "label": 'overlay', "value": 0}, { "label": 'difference', "value": 1}],
                                    ),
                                ]),

                                html.P(className="title is-size-6", children=["Map Variable:"]),
                                html.Div(className="subtitle", children=[
                                    dcc.Dropdown(
                                        id="variable-dropdown",
                                        style={"fontSize": "14px"},
                                        options=[{ "label": dn.detailsFiles[i]["label"], "value": i} for i in dn.detailsFiles.keys()],
                                    ),
                                ]),
                                html.P("Choose variables to show on the left and right plots."),
                                html.P(className="title is-size-6", children=["Left Plot Variable:"]),
                                html.Div(className="subtitle", children=[
                                    dcc.Dropdown(
                                        id="left-variable-dropdown",
                                        style={"fontSize": "14px"},
                                        options=[{ "label": dn.detailsVars[i]["label"], "value": i} for i in dn.detailsVars.keys()],
                                    ),
                                ]),
                                html.P(className="title is-size-6", children=["Right Plot Variable:"]),
                                html.Div(className="subtitle", children=[
                                    dcc.Dropdown(
                                        id="right-variable-dropdown",
                                        style={"fontSize": "14px"},
                                        options=[{ "label": dn.detailsVars[i]["label"], "value": i} for i in dn.detailsVars.keys()],
                                    ),
                                ]),
                            ])
                        ),
                        id="comparison-fade",
                        is_in=False,
                        appear=False,
                        className="mb-3",
                    ),
                    dbc.Button("Display Basic Info", id="basics-fade-button", className="mb-3"),
                    dbc.Fade(

                        dbc.Card(
                            dbc.CardBody(children=[
                                html.H4("Select data to show on the grid.", className="card-title"),
                                #html.P("Directions on how to use these tools", className="card-text"),

                                #html.P(className="title is-size-6", children=["Select Data to Show on Grid:"]),
                                html.Div(className="subtitle", children=[
                                    dcc.Dropdown(
                                        id="basics-dropdown",
                                        style={"fontSize": "14px"},
                                        options=[{ "label":"Maximal Wind Generation", "value": 0}],
                                    ),
                                ]),
                            ])
                        ),
                        id="basics-fade",
                        is_in=False,
                        appear=False,
                        className="mb-3",
                    ),
            ]),
            html.Div(className='column is-one-half', children=[
            # Right Vis tiles
            html.Div(className='tile is-parent is-vertical is-12', children=[
                # Slider tile
                html.Div(className='tile is-child', id="timeDiv", children=[
                    html.Div(className="tile", children=[
                        dbc.Button('Stop', id='stop-val', className="mr-1 mb-2", color="danger", size="sm", outline=True,n_clicks=0, disabled=True),
                        dbc.Button('Play', id='play-val', className="mr-1 mb-2", color="dark", size="sm", outline=True,n_clicks=0),
                        html.P(className='tag is-large', id="slider-val", children=[]),
                    ]),

                    html.Div(className="tile is-child ml-3 pl-6", children=[
                        dcc.Slider(
                            id='time-slider',
                            min = 0,
                            max = len(timeSteps)+10,
                            value = 0,
                            marks=dn.getMarks(timeSteps),
                            updatemode='drag',
                            className="sliderSmall ml-3 pl-3"
                        ),
                    ]),
                ]),
                # Map tile
                html.Div(className='tile is-child' , children=[
                    html.Div(className='tile is-dark mapDiv', children=[
                        dgl.DeckglLy(id='network-map',
                            mapboxtoken="pk.eyJ1Ijoia3BvdHRlciIsImEiOiJCNFlOLWVnIn0.IdEiAEoZbboAuuqOYtWg0w",
                            mapStyle="mapbox://styles/kpotter77/ck6ct8e2w0o1o1imrq0runboi",
                            viewState = viewState,
                            layers=initialLayers
                        ),
                      ])
                ]),
                # Graphs
                html.Div(className='tile is-child', id='plotsDiv', children=[
                    html.Article(className='tile', children=[
                        dcc.Graph(id="left-chart", config={'displayModeBar': False}),
                        dcc.Graph(id="right-chart", config={'displayModeBar': False}),
                    ])
                ])
            ]),
        ])
    ])
])

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
# Callbacks
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
@app.callback(Output("mode", "data"),
              Output("scenario-fade", "is_in"),
              Output("comparison-fade", "is_in"),
              Output("basics-fade", "is_in"),
              Input("scenario-fade-button", "n_clicks"),
              Input("comparison-fade-button", "n_clicks"),
              Input("basics-fade-button", "n_clicks"),
              State("scenario-fade", "is_in"),
              State("comparison-fade", "is_in"),
              State("basics-fade", "is_in"),
             )
def toggle_fade(scenario_n, comparison_n, basics_n, scenario_is_in, comparison_is_in, basics_is_in):

    # Change outputs based on which input is triggered
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id']

    mode = None
    scenario = scenario_is_in
    comparison = comparison_is_in
    basics = basics_is_in

    if(triggered=="."):
        mode = "scenario"

    elif(triggered == "scenario-fade-button.n_clicks"):
        basics = False
        scenario = not scenario
        comparison = not scenario
        mode =  "scenario" if scenario else "comparison"

    elif(triggered == "comparison-fade-button.n_clicks"):
        scenario = False
        comparison = not comparison
        basics = not comparison
        mode = "comparison" if comparison else "basics"

    elif(triggered == "basics-fade-button.n_clicks"):
        comparison = False
        basics = not basics
        scenario = not basics
        mode = "basics" if basics else "scenario"

    return mode, scenario, comparison, basics

#------------------------------------------------#
# Callback for playback
@app.callback(Output('play-val', 'disabled'),
              Output('stop-val', 'disabled'),
              Input('play-val', 'n_clicks_timestamp'),
              Input('stop-val', 'n_clicks_timestamp'),
              prevent_initial_call=True)
def useButtons(play_clicks, stop_clicks):
    if(stop_clicks == None):
        return True, False
    elif(play_clicks > stop_clicks):
        return True, False
    else:
        return False, True

#------------------------------------------------#
# Callback for time interval & time slider
@app.callback(Output('time-slider', 'value'),
              Input('time-interval', 'n_intervals'),
              State('play-val', 'disabled'), State('timestep', 'data'),
              prevent_initial_call=True)
def updateSlider(intervalTick, playVal, timestep):
    if(playVal == None and timestep==None):
        return 0
    else:
        if(playVal):
            return timestep + 1
        else:
            raise PreventUpdate()

#------------------------------------------------#
# Callbacks to update time slider
#------------------------------------------------#
@app.callback(Output('timestep', 'data'),
              Output('slider-val', 'children'),
              Input('time-slider', 'value'),
              prevent_initial_call=True)
def updateTimestep(value):
    if(value >= len(timeSteps)):
        return 0, ("   Timestep: "+pd.to_datetime(timeSteps[len(timeSteps)-1]).strftime("%m/%d, %H:%M"))
    else:
        return value, ("   Timestep: "+pd.to_datetime(timeSteps[value]).strftime("%m/%d, %H:%M"))

#------------------------------------------------#
# Callbacks to update the runs data
#------------------------------------------------#
@app.callback(Output('runsFile','data'),
              Output('summaryExtents', 'data'),
              Input('scenario-dropdown', 'value'),
              prevent_initial_call=True)
def updateRunsFile(directory):

    haveData = False

    # Read in the loss of load
    try:
        bdLoss = pd.read_csv(SCENARIO_PATH.joinpath(directory+"/"+"real_loss_of_load.csv")).T.reset_index()
        bdLoss.columns = bdLoss.iloc[0]
        bdLoss = bdLoss.rename(columns={"DateTime": 'Bus ID'})
        bdLoss = bdLoss.iloc[1:]
        gu.setBusLossData(bdLoss)
        haveData = True

    except IOError:
        gu.setBusLossData(None)
        haveData = False

    # Read in the thermal set points
    try:
        bdThermal = pd.read_csv(SCENARIO_PATH.joinpath(directory+"/"+"real_thermal_set_points_processed.csv")).T.reset_index()
        bdThermal.columns = bdThermal.iloc[0]
        bdThermal = bdThermal.rename(columns={"DateTime": 'Bus ID'})
        bdThermal = bdThermal.iloc[1:]
        gu.setBusThermalData(bdThermal)
        haveData = True

    except IOError:
        gu.setBusThermalData(None)
        haveData = False

    # Read in the renewable set points
    try:
        bdRenewable = pd.read_csv(SCENARIO_PATH.joinpath(directory+"/"+"real_renewable_setpoints_processed.csv")).T.reset_index()
        bdRenewable.columns = bdRenewable.iloc[0]
        bdRenewable = bdRenewable.rename(columns={"DateTime": 'Bus ID'})
        bdRenewable = bdRenewable.iloc[1:]
        gu.setBusRenewableData(bdRenewable)
        haveData = True
    except IOError:
        gu.setBusRenewableData(None)
        haveData = False

    # Read in the summary file
    try:
        summaryFile = pd.read_csv(SCENARIO_PATH.joinpath(dn.scenarioDirs[directory]['summary']))
        summaryFile = summaryFile[['DateTime','ActLossLoad','ActOverLoad','ActRenew','ActSpill','ActFirstStageCost','ActSecondStageCost']]
        gu.setSummaryData(summaryFile)
        summaryMinCost = summaryFile[['ActFirstStageCost','ActSecondStageCost']]
        summaryMinLoad = summaryFile[['ActLossLoad','ActOverLoad','ActRenew','ActSpill']]
        summaryExtents ={"cost":[summaryMinCost.values.min(),summaryMinCost.values.max()], "load":[summaryMinLoad.values.min(), 3448.505510]}
        haveData = True
    except IOError:
        gu.setSummaryData(None)
        haveData = False

    # Read in the actuals
    try:
        actuals = pd.read_csv(SCENARIO_PATH.joinpath("actuals.csv"))
        gu.setActualsData(actuals)
        haveData = True
    except IOError:
        gu.setActualsData(None)
        haveData = False

    return haveData, summaryExtents

#------------------------------------------------#
# Callbacks to update the basics data
#------------------------------------------------#
@app.callback(Output('basicsFile','data'),
              Input('basics-dropdown','value'),
              prevent_initial_call=True)
def updateBasicsFile(value):

    haveData = False

    # The get the maximal generation values
    if(value==0):
        try:
            maxFile = pd.read_csv(DATA_PATH.joinpath("RTS/maxGen.csv"))
            merged = bus.merge(maxFile, on='Bus ID', how='left').fillna(0)
            gu.setBasicsData(merged)
            haveData = True
        except IOError:
            gu.setBasicsData(None)
            haveData = False
    return haveData

#------------------------------------------------#
# Callbacks to update the basics data
#------------------------------------------------#
@app.callback(Output('compareFile','data'),
              Input('scenario-1-dropdown','value'),
              Input('scenario-2-dropdown','value'),
              Input('operator-dropdown', 'value'),
              Input('variable-dropdown', 'value'),
              Input('left-variable-dropdown', 'value'),
              Input('right-variable-dropdown', 'value'),
              prevent_initial_call=True)
def updateCompareFile(scenario1, scenario2, operator, mapVar, leftVar, rightVar):

    haveData = False

    if(scenario1 != None) and (scenario2 != None) and (operator != None) and (mapVar != None) and (leftVar != None and rightVar != None):

        try:
            # Read in the two  runs
            s1 = pd.read_csv(SCENARIO_PATH.joinpath(scenario1+"/"+mapVar)).T.reset_index()
            s2 = pd.read_csv(SCENARIO_PATH.joinpath(scenario2+"/"+mapVar)).T.reset_index()

            # Reset the indexes
            s1.columns = s1.iloc[0]
            s1 = s1.iloc[1:]
            s1 = s1.rename(columns={"DateTime": 'Bus ID'})
            s1.set_index('Bus ID', inplace=True, drop=True)

            # Reset the indexes
            s2.columns = s2.iloc[0]
            s2 = s2.iloc[1:]
            s2 = s2.rename(columns={"DateTime": 'Bus ID'})
            s2.set_index('Bus ID', inplace=True, drop=True)

            # If we have the overlay operator
            if(operator == 0):
                gu.setCompareData({'type':'overlay',
                                   'data': {"s1":s1.reset_index(level=0, inplace=False),
                                            "s2":s2.reset_index(level=0, inplace=False)}})
            # If we have a difference operator
            elif(operator == 1):
                # Get the absolute value of the difference
                difference = s1-s2
                difference = difference.abs()
                difference.reset_index(level=0, inplace=True)
                gu.setCompareData({'type':'difference', 'data': difference})


            # Get the plot files
            summaryFile1 = pd.read_csv(SCENARIO_PATH.joinpath(dn.scenarioDirs[scenario1]['summary']))
            summaryFile2 = pd.read_csv(SCENARIO_PATH.joinpath(dn.scenarioDirs[scenario2]['summary']))
            leftData1 = summaryFile1[['DateTime',leftVar]]
            leftData2 = summaryFile2[['DateTime',leftVar]]

            lExtent = [leftData1[leftVar].values.min() if leftData1[leftVar].values.min() < leftData2[leftVar].values.min() else leftData2[leftVar].values.min(),
                       leftData1[leftVar].values.max() if leftData1[leftVar].values.max() > leftData2[leftVar].values.max() else leftData2[leftVar].values.max()]

            gu.setLeftPlotData({"one":leftData1, "two":leftData2,'tag':leftVar, 'oneScen':scenario1,'twoScen':scenario2, 'extents':lExtent})
            rightData1 = summaryFile1[['DateTime',rightVar]]
            rightData2 = summaryFile2[['DateTime',rightVar]]

            r1Min = rightData1[rightVar].values.min()
            r2Min = rightData2[rightVar].values.min()
            r1Max = rightData1[rightVar].values.max()
            r2Max = rightData2[rightVar].values.max()

            rExtent = [r1Min if r1Min < r2Min else r2Min, r1Max if r1Max > r2Max else r2Max]

            gu.setRightPlotData({"one":rightData1, "two":rightData2, 'tag':rightVar, 'oneScen':scenario1,'twoScen':scenario2, 'extents':rExtent})

            haveData = True
        except IOError:
            gu.setCompareData(None)
            gu.setLeftPlotData(None)
            gu.setRightPlotData(None)
            haveData = False


    if(haveData == False):
        raise PreventUpdate()
    else:
        return haveData

#------------------------------------------------#
# Callbacks for the summary/timeseries files
#------------------------------------------------#
@app.callback(Output('left-chart', 'figure'),
              Output('right-chart', 'figure'),
              Output('timeDiv', 'style'),
              Output('plotsDiv', 'style'),
              Input('timestep', 'data'),
              Input('runsFile', 'data'),
              Input('basicsFile', 'data'),
              Input('compareFile', 'data'),
              Input('mode', 'data'),
              State('mode', 'data'),
              State('summaryExtents', 'data'),
              prevent_initial_call=True)
def createCharts(timestep, runsFile, compareFile, basicsFile, showMode, modeState, summaryExtents):

    # Change outputs based on which input is triggered
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id']

    # Return values
    leftFig = dash.no_update
    rightFig = dash.no_update
    timeDivStyle = {"visibility":'visible'}
    plotsDivStyle = {"visibility":'visible'}

    # Create the runs charts
    if(showMode == 'scenario'):

        # Get the globally stored files
        summaryFile = gl.summaryStore
        actuals = gl.actualsStore

        if(summaryFile is not None) and (actuals is not None) and (summaryExtents is not None):

            # Cost figure
            df = summaryFile
            date = df['DateTime']
            leftFig = go.Figure()
            leftFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActFirstStageCost'], fill='tozeroy', name='ActFirstStageCost', stackgroup='one',line=dict(color="#008B8B")))
            leftFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActSecondStageCost'], fill='tonexty',name='ActSecondStageCost', stackgroup='one',line=dict(color="#F08080")))
            leftFig.add_trace(go.Scatter(x=[daterange[timestep], daterange[timestep]], y=[summaryExtents['cost'][0], 110000], name='Time Step', line=dict(color="#2F4F4F")))

            # Actuals figure
            rightFig = go.Figure()
            rightFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActLossLoad'], fill='tozeroy', name='Actual Loss Load', stackgroup='one',line=dict(color=dn.lossHex)))
            rightFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActSpill'], fill='tonexty',name='Actual Spill', stackgroup='one',line=dict(color='#FF7F50')))
            rightFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActRenew'], fill='tonexty',name='Actual Renewable', stackgroup='one',line=dict(color=dn.renewableHex)))
            rightFig.add_trace(go.Scatter(x=actuals['DateTime'], y=actuals['total'],name='Actual Wind', line=dict(color='royalblue', width=4, dash='dot')))
            rightFig.add_trace(go.Scatter(x=[daterange[timestep], daterange[timestep]], y=[summaryExtents['load'][0],4500], name='Time Step', line=dict(color="#2F4F4F")))

            leftFig.update_layout(
                autosize=False,
                width=700,
                height=300,
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=0,
                    pad=0
                ),
                legend=dict(
                    yanchor="bottom",
                    y=.65,
                    xanchor="right",
                    x=0.99
                )
            )
            rightFig.update_layout(
                autosize=False,
                width=700,
                height=300,
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=0,
                    pad=0
                ),
                legend=dict(
                    yanchor="bottom",
                    y=.5,
                    xanchor="right",
                    x=0.99
                )
            )
            leftFig.update_xaxes(
                title_text = "Time",
                range=[daterange[0],daterange[len(daterange)-1]],
                showgrid= False,
            )
            leftFig.update_yaxes(
                title_text = "Cost ($/minute)",
                showgrid= False,
                range=[0, 120000],
                title_font=dict(size=10),
            )
            rightFig.update_xaxes(
                title_text = "Time",
                range=[daterange[0],daterange[len(daterange)-1]],
                showgrid = False)
            rightFig.update_yaxes(
                title_text = "Capacity kW",
                showgrid= False,
                range=[0, 5000],
                title_font=dict(size=10),
            )


    # Hide the timestep slider and charts
    elif(showMode == 'basics'):
        timeDivStyle = {"visibility":'hidden'}
        plotsDivStyle = {"visibility":'hidden'}

    # Create the compare charts
    elif(showMode == "comparison"):
        leftData = gl.leftPlotStore
        rightData = gl.rightPlotStore

        if(leftData is not None) and (rightData is not None):

            leftData1 = leftData['one']
            leftData2 = leftData['two']
            lExtents = leftData['extents']
            lTag = leftData['tag']
            scn1 = leftData['oneScen']
            scn2 = leftData['twoScen']
            s1Name = dn.scenarioDirs[scn1]['label'].split("(")[0]
            s2Name = dn.scenarioDirs[scn2]['label'].split("(")[0]

            leftFig = go.Figure()
            leftFig.add_trace(go.Scatter(x=leftData1['DateTime'], y=leftData1[lTag], fill='tozeroy', name=s1Name,line=dict(color="#008B8B")))
            leftFig.add_trace(go.Scatter(x=leftData2['DateTime'], y=leftData2[lTag], name=s2Name,line=dict(color="#F08080")))
            leftFig.add_trace(go.Scatter(x=[daterange[timestep], daterange[timestep]], y=[lExtents[0], lExtents[1]], name='Time Step', line=dict(color="#2F4F4F")))

            rightData1 = rightData['one']
            rightData2 = rightData['two']
            rTag = rightData['tag']
            rExtents = rightData['extents']

            rightFig = go.Figure()
            rightFig.add_trace(go.Scatter(x=rightData1['DateTime'], y=rightData1[rTag], fill='tozeroy', name=s1Name, line=dict(color="#008B8B")))
            rightFig.add_trace(go.Scatter(x=rightData2['DateTime'], y=rightData2[rTag], name=s2Name, line=dict(color="#F08080")))
            rightFig.add_trace(go.Scatter(x=[daterange[timestep], daterange[timestep]], y=[rExtents[0], rExtents[1]], name='Time Step', line=dict(color="#2F4F4F")))

            leftFig.update_layout(
                title={
                    'text': dn.detailsVars[lTag]['label'],
                    'y':0.9,
                    'x':0.15,
                    'xanchor': 'left',
                    'yanchor': 'top',
                },
                autosize=False,
                width=700,
                height=300,
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=50,
                    pad=0
                ),
                legend=dict(
                    yanchor="bottom",
                    y=1,
                    xanchor="right",
                    x=1
                )
            )
            rightFig.update_layout(
                title={
                    'text': dn.detailsVars[rTag]['label'],
                    'y':0.9,
                    'x':0.15,
                    'xanchor': 'left',
                    'yanchor': 'top',
                },
                autosize=False ,
                width=700,
                height=300,
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=50,
                    pad=0
                ),
                legend=dict(
                    yanchor="bottom",
                    y=1,
                    xanchor="right",
                    x=1
                )
            )
            leftFig.update_xaxes(
                title_text = "Time",
                range=[daterange[0],daterange[len(daterange)-1]],
                showgrid= False,
            )
            leftFig.update_yaxes(
                title_text = dn.detailsVars[lTag]['label'],
                showgrid= False,
                #range=[0, 120000],
                title_font=dict(size=10),
            )
            rightFig.update_xaxes(
                title_text = "Time",
                range=[daterange[0],daterange[len(daterange)-1]],
                showgrid = False)
            rightFig.update_yaxes(
                title_text = dn.detailsVars[rTag]['label'],
                showgrid= False,
                #range=[0, 5000],
                title_font=dict(size=10),
            )


    return leftFig, rightFig, timeDivStyle, plotsDivStyle

#------------------------------------------------#
# Callbacks for detail files
#------------------------------------------------#
@app.callback(Output("network-map", "layers"),
              Input('timestep', 'data'),
              Input('runsFile', 'data'),
              Input('basicsFile', 'data'),
              Input('compareFile', 'data'),
              Input('mode', 'data'),
              State('mode', 'data'),
              prevent_initial_call=True)
def createMap(timestep, runsFile, basicsFile, compareFile, showMode, modeState):

    # Change outputs based on which input is triggered
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id']

    # Create the map layers
    mapLayers=[]

    # If we are showing default runs
    if(showMode == "scenario"):
        # If we have bus data
        if(runsFile == True):

            # Add the branch layer
            mapLayers.append(initialLayers[0])
            mapLayers.append(initialLayers[1])

            # Grab the loss of load
            busLoss = gl.busLossStore[['Bus ID',daterange[timestep]]]
            busLoss = busLoss.astype({'Bus ID': 'int64', daterange[timestep]:'float64'})
            busLoss = busLoss.rename(columns={daterange[timestep]: "Loss"})

            # Grab the thermal generation
            busThermal= gl.busThermalStore[['Bus ID',daterange[timestep]]]
            busThermal = busThermal.astype({'Bus ID': 'int64', daterange[timestep]:'float64'})
            busThermal = busThermal.rename(columns={daterange[timestep]: "Thermal"})

            # Grab the renewable generation
            busRenewable = gl.busRenewableStore[['Bus ID',daterange[timestep]]]
            busRenewable = busRenewable.astype({'Bus ID': 'int64', daterange[timestep]:'float64'})
            busRenewable = busRenewable.rename(columns={daterange[timestep]: "Renewable"})

            # Append the data to the geometry data
            busMapData = pd.merge(busLoss, bus, on='Bus ID')
            busMapData = pd.merge(busThermal, busMapData, on='Bus ID')
            busMapData = pd.merge(busRenewable, busMapData, on='Bus ID')

            # Create the reneweable layer
            renewableScatterLayer = {
                "type": "ScatterplotLayer",
                "id": 'scatterplot-layer-renewable',
                "data": busMapData.to_dict(orient='records'),
                "pickable": False,
                "opacity": .25,
                "stroked": True,
                "filled": True,
                "radiusScale": 50,
                "lineWidthScale": 25,
                "radiusMinPixels": 0,
                "lineWidthMinPixels": 0,
                "getPosition": "function(d){return [d['lng'], d['lat']]}",
                "getRadius": "function(d){return d['Renewable']}",
                "getLineWidth": "function(d){return d['Renewable']}",
                "getLineColor": dn.renewableColor,
                "getFillColor": dn.renewableColor,
            }
            mapLayers.append(renewableScatterLayer)

            # Create the thermal layer
            thermalScatterLayer = {
                "type": "ScatterplotLayer",
                "id": 'scatterplot-layer-thermal',
                "data": busMapData.to_dict(orient='records'),
                "pickable": False,
                "opacity": .25,
                "stroked": True,
                "filled": True,
                "radiusScale": 50,
                "lineWidthScale": 25,
                "radiusMinPixels": 0,
                "lineWidthMinPixels": 0,
                "getPosition": "function(d){return [d['lng'], d['lat']]}",
                "getRadius": "function(d){return d['Thermal']}",
                "getLineWidth": "function(d){return d['Thermal']}",
                "getLineColor": dn.thermalColor,
                "getFillColor": dn.thermalColor,
            }
            mapLayers.append(thermalScatterLayer)

            # Create the loss layer
            lossScatterLayer = {
                "type": "ScatterplotLayer",
                "id": 'scatterplot-layer2',
                "data": busMapData.to_dict(orient='records'),
                "pickable": False,
                "opacity": 0.5,
                "stroked": True,
                "filled": True,
                "radiusScale":50,
                "lineWidthScale": 25,
                "radiusMinPixels": 0,
                "lineWidthMinPixels": 0,
                "getPosition": "function(d){return [d['lng'], d['lat']]}",
                "getRadius": "function(d){return d['Loss']}",
                "getLineWidth": "function(d){return d['Loss']}",
                "getLineColor": "function(d){ let color = d['Loss']==0 ? [248, 248, 255] : [227,26,28]; return color}",
                "getFillColor": "function(d){ let color = d['Loss']==0 ? [248, 248, 255] : [227,26,28]; return color}",
            }
            mapLayers.append(lossScatterLayer)

    elif(showMode == "comparison"):
        mapLayers.append(initialLayers[0])
        mapLayers.append(initialLayers[1])
        if(compareFile == True):

            # Create the difference scatter layer
            if(gl.compareStore['type'] == "difference"):
                diff = gl.compareStore['data'][['Bus ID',daterange[timestep]]]
                diff = diff.astype({'Bus ID': 'int64', daterange[timestep]:'float64'})
                diff = diff.rename(columns={daterange[timestep]: "diff"})
                diffMap = pd.merge(diff, bus, on='Bus ID')

                # Create the diff layer
                diffScatterLayer = {
                    "type": "ScatterplotLayer",
                    "id": 'scatterplot-layer-renewable',
                    "data": diffMap.to_dict(orient='records'),
                    "pickable": False,
                    "opacity": .5,
                    "stroked": True,
                    "filled": True,
                    "radiusScale": 50,
                    #"lineWidthScale": 25,
                    "radiusMinPixels": 1,
                    "lineWidthMinPixels": 1,
                    "getPosition": "function(d){return [d['lng'], d['lat']]}",
                    "getRadius": "function(d){return d['diff']}",
                    #"getLineWidth": "function(d){return d['diff']}",
                    "getLineColor": dn.lineColor,
                    "getFillColor": dn.diffColor,
                }
                mapLayers.append(diffScatterLayer)

            elif(gl.compareStore['type'] == "overlay"):
                r1 = gl.compareStore['data']['s1'][['Bus ID',daterange[timestep]]]
                r1 = r1.astype({'Bus ID': 'int64', daterange[timestep]:'float64'})
                r1 = r1.rename(columns={daterange[timestep]: "diff"})
                r1 = pd.merge(r1, bus, on='Bus ID')

                r2 = gl.compareStore['data']['s2'][['Bus ID',daterange[timestep]]]
                r2 = r2.astype({'Bus ID': 'int64', daterange[timestep]:'float64'})
                r2 = r2.rename(columns={daterange[timestep]: "diff"})
                r2 = pd.merge(r2, bus, on='Bus ID')

                # Create the r1 layer
                r1ScatterLayer = {
                    "type": "ScatterplotLayer",
                    "id": 'scatterplot-layer-renewable',
                    "data":r1.to_dict(orient='records'),
                    "pickable": False,
                    "opacity": .35,
                    "stroked": True,
                    "filled": True,
                    "radiusScale": 50,
                    "radiusMinPixels": 0,
                    "lineWidthMinPixels": 2,
                    "getPosition": "function(d){return [d['lng'], d['lat']]}",
                    "getRadius": "function(d){return d['diff']}",
                    #"getLineWidth": "function(d){return d['diff']}",
                    "getLineColor": dn.r1Color,
                    "getFillColor": dn.r1Color,
                }
                mapLayers.append(r1ScatterLayer)
                r2ScatterLayer = {
                    "type": "ScatterplotLayer",
                    "id": 'scatterplot-layer-renewable',
                    "data":r2.to_dict(orient='records'),
                    "pickable": False,
                    "opacity": .75,
                    "stroked": True,
                    "filled": False,
                    "radiusScale": 50,
                    "radiusMinPixels": 0,
                    "lineWidthMinPixels": 3,
                    "getPosition": "function(d){return [d['lng'], d['lat']]}",
                    "getRadius": "function(d){return d['diff']}",
                    #"getLineWidth": "function(d){return d['diff']}",
                    "getLineColor": dn.r2Color,
                    "getFillColor": dn.r2Color,
                }
                mapLayers.append(r2ScatterLayer)

    elif(showMode == "basics"):

        # Add the branch layer
        mapLayers.append(initialLayers[0])

        # Create the basics layer
        if(basicsFile == True):
            basics = gl.basicsStore
            scatterLayer = {
                "type": "ScatterplotLayer",
                "id": 'scatterplot-layer2',
                "data": basics.to_dict(orient='records'),
                "pickable": False,
                "opacity": 1.0,
                "stroked": True,
                "filled": True,
                "radiusScale":15,
                "radiusMinPixels":2,
                "lineWidthMinPixels": 1,
                "getPosition": "function(d){return [d['lng'], d['lat']]}",
                "getRadius": "function(d){return d['GenMWMax']}",
                "getLineWidth":2,
                "getLineColor": [47, 79, 79],
                "getFillColor":"function(d){ let color = d['GenMWMax']==0 ? [31,120,180] : [178,223,138]; return color}",
            }
            mapLayers.append(scatterLayer)
        else:
            # Or just add the default buses
            mapLayers.append(initialLayers[1])



    return mapLayers


if __name__ == '__main__':
    app.run_server(debug=True)

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

# ----------------------------------------------------------- #
# Read in the data and do some initial processing

# Set the path to the data
DATA_PATH = pathlib.Path(__file__).parent.joinpath("./data").resolve()
SCENARIO_PATH = DATA_PATH.joinpath("50Scen_3days 2")

print("DATAPATH", DATA_PATH)

# Read in the network
bus = pd.read_csv(DATA_PATH.joinpath("RTS/bus.csv"))
bus = bus[['Bus ID','lat','lng']]
branch = pd.read_csv(DATA_PATH.joinpath("RTS/branch.csv"))
branch = branch[['UID','From Bus','To Bus']]
#gen = pd.read_csv(DATA_PATH.joinpath("RTS/gen.csv"))

#print(gen)
#sys.exit()

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
# for d in timeSteps:
#     splitT = d.split("T")
#     split = splitT[1].split(":")
#     time = splitT[0]+" "+split[0]+":"+split[1]
#     daterange.append(time)
#

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

# ----------------------------------------------------------- #
# Define labels and colors
scenarioDirs = {'details50_AC_72hrs':{"label":"AC OPF", "summary":"ac_results.csv"},
                'details50_DC_72hrs':{"label":"DC OPF", "summary":"dc_results.csv"},
                'details50_CP_72hrs':{"label":"Copper Plate OPF (Real power only)", "summary":"cp_results.csv"},
                'details50_ACCP_72hrs':{"label":"Multi-Fidelity AC-CP OPF", "summary":"mf_ACCP_results.csv"},
                'details50_DCCP_72hrs':{"label":"Multi-Fidelity DC-CP OPF", "summary":"mf_DCCP_results.csv"},
                'details50_cp_ac_conseq':{"label":"Copper Plate with AC recourse (Includes reactive power)", "summary":"cp_ac_actuals_results.csv"},
                'details50_cp_dc_conseq':{"label":"Copper Plate with DC recourse (Real power only)", "summary":"cp_dc_actuals_results.csv"}}
busVariableFiles = {'real_loss_of_load.csv':{'title':'Real Loss of Load','color':'whiteRed', 'process':False},
                    'real_thermal_set_points_processed.csv':{'title':'Real Thermal Set Points','color':'whiteRed', 'process':True},
                    'real_renewable_setpoints_processed.csv':{'title':'Real Renewable Setpoints','color':'whiteRed', 'process':True},

                    # Possibly of interest later
                    #'real_bus_voltage_angle.csv':{'title':'Real Bus Voltage Angle', 'color':'whiteRed', 'process':False},
                    #'real_bus_voltage_magnitude.csv':{'title':'Real Bus Voltage Magnitude', 'color':'whiteRed', 'process':False},
                    #'reactive_thermal_set_points.csv':{'title':'Reactive Thermal Set Points','color':'whiteRed', 'process':True},
                    #'real_renewable_spill.csv':{'title':'Real Renewable Spill','color':'whiteRed', 'process':False},

                    # Zeros
                    #'reactive_loss_of_load.csv':{'title':'Reactive Loss of Load','color':'whiteRed', 'process':False},
                    #'real_overload.csv':{'title':'Real Overload','color':'whiteRed', 'process':False},
                    #'reactive_overload.csv':{'title':'Reactive Overload','color':'whiteRed', 'process':False},
                    # Debugging (ignore)
                    #'expected_Pth.csv':{'title':'Expected Pth','color':'cvidis', 'process':'false'},

                  }
branchVariableFiles ={'real_branch_power_flow.csv':{'title':'Real Branch Power Flow','color':'whiteRed'}}

# Color ranges for the different variables
colorMaps = {
    "whiteRed":{
        "range": [[248, 248, 255],[255,69,0]],
        "rangeRGB": ['rgb(248,248,255)','rgb(255, 69, 0)'],
        "scale": "linear"
    },
    "blackRed":{
        "range": [[0, 0, 0],[255,69,0]],
        "rangeRGB": ['rgb(0,0,0)','rgb(255, 69, 0)'],
        "scale": "linear"
    },
    "tealBlu":{
        "range": [[127, 255, 212],[72, 61, 139]],
        "rangeRGB": ['rgb(127, 255, 212)','rgb(72, 61, 139)'],
        "scale": "linear"
    },
    "blu":{
        "range": [[230, 247, 255],[0, 85, 128]],
        "rangeRGB": ['rgb(230, 247, 255)','rgb(0, 85, 128)'],
        "scale": "linear"
    },
    "grn":{
        "range": [[236, 249, 242],[45, 134, 89]],
        "rangeRGB": ['rgb(236, 249, 242)','rgb(45, 134, 89)'],
        "scale": "linear"
    },
    "geojson" : {
        "range": [[166,97,26],[223,194,125],[255,255,255],[128,205,193],[1,133,113]],
        "rangeRGB": ['rgb(166,97,26)','rgb(223,194,125)','rgb(255,255,255)','rgb(128,205,193)','rgb(1,133,113)'],
        "scale": "diverging"
    },
    "ploygon": {
        "range": [[255,255,212],[254,227,145],[254,196,79],[254,153,41],[236,112,20],[204,76,2],[140,45,4]],
        "rangeRGB": ['rgb(255,255,212)','rgb(254,227,145)','rgb(254,196,79)','rgb(254,153,41)','rgb(236,112,20)','rgb(204,76,2)','rgb(140,45,4)'],
        "scale" : "quantize"
    },
    "bus":{
        "range": [[0,255,0],[254,0,0]],
        "rangeRGB": ['rgb(0,255,0)','rgb(254,0,0)'],
        "scale" : "linear"
    },
    "branch":{
        "range": [[255,0,0],[0,0,255]],
        "rangeRGB": ['rgb(255,0,0)','rgb(0,0,255)'],
        "scale" : "linear"
    }
}

renewableColor = [51,160,44]
renewableHex ='#33A02C'
thermalColor = [31,120,180]
thermalHex = '#1F78B4'
lossColor = [227,26,28]
lossHex = '#E31A1C'


# ----------------------------------------------------------- #
# Time helper functions
def unixTimeMillis(dt):
    ''' Convert datetime to unix timestamp '''
    return int(time.mktime(dt.timetuple()))
def unixToDatetime(unix):
    ''' Convert unix timestamp to datetime. '''
    return pd.to_datetime(unix,unit='s')

# Returns the marks for labeling the slider
def getMarks():
    result = {}
    for i in range(0, len(timeSteps), 216):
        result[i] = pd.to_datetime(timeSteps[i]).strftime("%D:%H:%M:%S")
    return result

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
# Layout
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#

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
        html.Div(className='section', children=[
            html.Div(className='tile is-ancestor', children=[
                # Left UI tiles
                #html.Div(className='tile is-parent is-vertical is-2 uiDivs', children=[

                #]),
                # Right Vis tiles
                html.Div(className='tile is-parent is-vertical is-12', children=[
                    #html.Article(className='tile is-child', children=[
                        html.P(className="title is-size-6", children=["Power Flow Scenario:"]),
                        html.Div(className="subtitle", children=[
                            dcc.Dropdown(
                                id="scenario-dropdown",
                                style={"fontSize": "14px"},
                                options=[{ "label": scenarioDirs[i]["label"], "value": i} for i in scenarioDirs.keys()],
                            ),
                        ]),
                        # html.P(className="title is-size-6", children=["Bus Variable:"]),
                        # html.Div(className="subtitle", children=[
                        #     dcc.Dropdown(
                        #         id="bus-variable-dropdown",
                        #         style={"fontSize": "14px"},
                        #     ),
                        # ]),
                        # html.P(className="title is-size-6", children=["Branch Variable:"]),
                        # html.Div(className="subtitle", children=[
                        #     dcc.Dropdown(
                        #         id="branch-variable-dropdown",
                        #         style={"fontSize": "14px"},
                        #     ),
                        # ]),
                    #]),
                    # Slider tile
                    html.Div(className='tile is-child', children=[
                        html.Div(className="tile is-child", children=[
                            html.Button('Stop', id='stop-val', n_clicks=0, disabled=True),
                            html.Button('Play', id='play-val', n_clicks=0),
                            html.P(id="slider-val", children=[])
                        ]),
                        html.Div(className="tile is-child", children=[
                            dcc.Slider(
                                id='time-slider',
                                min = 0,
                                max = len(timeSteps),
                                value = 0,
                                marks=getMarks(),
                                updatemode='drag'
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
                            # dgl.MapLegend(id='mapLegend',
                            #    title='Map Legend'
                            # )
                          ])
                    ]),
                    # Feeder graph tile
                    html.Div(className='tile is-child', children=[
                        html.Article(className='tile', children=[
                            dcc.Graph(id="timeseries-cost",
                                       config={'displayModeBar': False}),
                            #dcc.Graph(id="timeseries-cost-2",
                            #          config={'displayModeBar': False}),
                            dcc.Graph(id="timeseries-load",
                                       config={'displayModeBar': False}),
                        ])

                    ])
                ]),
            ]),

        ])
 ])

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
# Callbacks
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#

#------------------------------------------------#
# Callback for playback
@app.callback([Output('play-val', 'disabled'),
               Output('stop-val', 'disabled')],
              [Input('play-val', 'n_clicks_timestamp'),
               Input('stop-val', 'n_clicks_timestamp')],
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
              [Input('time-interval', 'n_intervals')],
              [State('play-val', 'disabled'), State('timestep', 'data')],
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
@app.callback([Output('timestep', 'data'),
               Output('slider-val', 'children')],
              [Input('time-slider', 'value')],
              prevent_initial_call=True)
def updateTimestep(value):
    if(value >= len(timeSteps)):
        return 0, ("   Timestep: "+str(timeSteps[len(timeSteps)-1]))
    else:
        return value, ("   Timestep: "+str(timeSteps[value]))

#------------------------------------------------#
# Callbacks to update the scenario directory
#------------------------------------------------#
@app.callback(Output('directory', 'data'),
               #Output('bus-variable-dropdown', 'options'),
               #Output('branch-variable-dropdown', 'options'),
               #Output('branch-variable-dropdown', 'disabled'),
               #Output('branch-variable-dropdown', 'placeholder'),],
              [Input('scenario-dropdown', 'value')],
              prevent_initial_call=True)
def updateScenarioDirectory(value):
    if(value == None):
        raise PreventUpdate()

    branchDisabled = False
    branchPlaceholder = "Select..."

    # Get the list of data sets in this directory
    dataFiles = glob.glob(str(SCENARIO_PATH.joinpath(value+"/*.csv")))

    dataFiles.sort()
    res = list(map(lambda st: str.replace(st, str(SCENARIO_PATH.joinpath(value+"/")), ""), dataFiles))
    res = list(map(lambda st: str.replace(st, "/", ""), res))

    # Find the bus or branch files that exist in the directory
    busFiles = []
    for i in res :
        if(i in busVariableFiles):
            busFiles.append(i)
    branchFiles = []
    for i in res :
        if(i in branchVariableFiles):
            branchFiles.append(i)

    busOptions=[{ "label": busVariableFiles[i]['title'], "value": i } for i in busFiles]
    branchOptions= [{ "label": branchVariableFiles[i]['title'], "value": i } for i in branchFiles]
    if(len(branchOptions) == 0):
        branchOptions =[{"label": "None", "value": 0}]
        branchDisabled = True
        branchPlaceholder = "None"

    return value#, busOptions, branchOptions, branchDisabled, branchPlaceholder


#------------------------------------------------#
# Callbacks to update the bus file
#------------------------------------------------#
@app.callback([Output('busFile','data'),
               Output('busLossExtents','data'),
               Output('busThermalExtents','data'),
               Output('busRenewableExtents','data'),
               Output('summaryExtents', 'data')],
              #[Input('bus-variable-dropdown', 'value')],
              [Input('directory', 'data')],
              prevent_initial_call=True)
def updateBusFile(directory):

    # Read in the data
    #if(busFile != None):
    # Read in the bus data, transpose and rename column headers
    # Read in the loss of load

    try:
        bdLoss = pd.read_csv(SCENARIO_PATH.joinpath(directory+"/"+"real_loss_of_load.csv")).T.reset_index()
        bdLoss.columns = bdLoss.iloc[0]
        bdLoss = bdLoss.rename(columns={"DateTime": 'Bus ID'})
        bdLoss = bdLoss.iloc[1:]
        gu.setBusLossData(bdLoss)
        lossMin = bdLoss.loc[ : , bdLoss.columns != 'Bus ID']
        busLossExtents = [lossMin.values.min(), lossMin.values.max()]
    except IOError:
        gu.setBusLossData(None)
        busLossExtents = [0,1]

    # Read in the bus data, transpose and rename column headers
    # Read in the thermal set points
    try:
        bdThermal = pd.read_csv(SCENARIO_PATH.joinpath(directory+"/"+"real_thermal_set_points_processed.csv")).T.reset_index()
        bdThermal.columns = bdThermal.iloc[0]
        bdThermal = bdThermal.rename(columns={"DateTime": 'Bus ID'})
        bdThermal = bdThermal.iloc[1:]
        gu.setBusThermalData(bdThermal)
        thermalMin = bdThermal.loc[ : , bdThermal.columns != 'Bus ID']
        busThermalExtents = [thermalMin.values.min(), thermalMin.values.max()]
        #print("Thermal", bdThermal.values.min(), bdThermal.values.max())
    except IOError:
        gu.setBusThermalData(None)
        busThermalExtents = [0,1]

    # Read in the bus data, transpose and rename column headers
    # Read in the renewable set points
    bdRenewable = pd.read_csv(SCENARIO_PATH.joinpath(directory+"/"+"real_renewable_setpoints_processed.csv")).T.reset_index()
    bdRenewable.columns = bdRenewable.iloc[0]
    bdRenewable = bdRenewable.rename(columns={"DateTime": 'Bus ID'})
    bdRenewable = bdRenewable.iloc[1:]
    gu.setBusRenewableData(bdRenewable)
    renewableMin = bdRenewable.loc[ : , bdRenewable.columns != 'Bus ID']
    busRenewableExtents = [renewableMin.values.min(), renewableMin.values.max()]

    # Read in the summary file
    summaryFile = pd.read_csv(SCENARIO_PATH.joinpath(scenarioDirs[directory]['summary']))
    summaryFile = summaryFile[['DateTime','ActLossLoad','ActOverLoad','ActRenew','ActSpill','ActFirstStageCost','ActSecondStageCost']]
    gu.setSummaryData(summaryFile)
    summaryMinCost = summaryFile[['ActFirstStageCost','ActSecondStageCost']]
    summaryMinLoad = summaryFile[['ActLossLoad','ActOverLoad','ActRenew','ActSpill']]
    #print(summaryMinCost.max())
    summaryExtents ={"cost":[summaryMinCost.values.min(),summaryMinCost.values.max()], "load":[summaryMinLoad.values.min(), 3448.505510]}

    # Read in the actuals
    actuals = pd.read_csv(SCENARIO_PATH.joinpath("actuals.csv"))
    #print(actuals)
    gu.setActualsData(actuals)


    return True, busLossExtents, busThermalExtents, busRenewableExtents, summaryExtents#, True, True#busVariableFiles[busFile]['title'], busVariableFiles[busFile]['color']


#------------------------------------------------#
# Callbacks to update the branch file
#------------------------------------------------#
# @app.callback([Output('branchFile', 'data'),
#                Output('branchVariable', 'data'),
#                Output('branchDataColor', 'data')],
#               [Input('branch-variable-dropdown', 'value')],
#               [State('directory', 'data')],
#               prevent_initial_call=True)
# def updateBranchFile(branchFile, directory):
#     # Read in the data
#     if(branchFile != None):
#         gu.setBranchData(pd.read_csv(SCENARIO_PATH.joinpath(directory+"/"+branchFile), dtype='float'))
#         return True, branchVariableFiles[branchFile]['title'], branchVariableFiles[branchFile]['color']
#     else:
#         return False, None, None

#------------------------------------------------#
# Callbacks for the summary/timeseries files
#------------------------------------------------#
@app.callback([Output('timeseries-cost', 'figure'),
               #Output("timeseries-cost-2", 'figure'),
               Output('timeseries-load', 'figure')],
              [Input('summaryExtents', 'data'),
               Input('timestep', 'data')],
              prevent_initial_call=True)
def createTimeSeriesSummaries(value, timestep):


    if(value == None):
        raise PreventUpdate()

    summaryFile = gl.summaryStore
    actuals = gl.actualsStore

    if(value != None):

        # Read in the summary file
        df = summaryFile
        date = df['DateTime']
        costFig = go.Figure()  #=make_subplots(rows=2, cols=1,  shared_xaxes=True)#), subplot_titles=("Exp Cost", "Act Cost", "Plot 3", "Plot 4"))

        costFig2 = go.Figure()
        # Exp cost
        #fig.add_trace(go.Scatter(x=df['DateTime'], y=df['ExpFirstStageCost'], fill='tozeroy', name='ExpFirstStageCost', stackgroup='one', ),  row=1, col=1)
        #fig.add_trace(go.Scatter(x=df['DateTime'], y=df['ExpSecondStageCost'], fill='tonexty', name='ExpSecondStageCost',  stackgroup='one'),  row=1, col=1)
        # Act cost
        costFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActFirstStageCost'], fill='tozeroy', name='ActFirstStageCost', stackgroup='one',line=dict(color="#008B8B")))
        costFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActSecondStageCost'], fill='tonexty',name='ActSecondStageCost', stackgroup='one',line=dict(color="#F08080")))
        costFig.add_trace(go.Scatter(x=[daterange[timestep], daterange[timestep]], y=[value['cost'][0], 110000], name='Time Step', line=dict(color="#2F4F4F")))
        # Exp vars
        #fig.add_trace(go.Scatter(x=df['DateTime'], y=df['ExpLossLoad'], fill='tozeroy', name='ExpLossLoad', stackgroup='one',),  row=3, col=1)
        #fig.add_trace(go.Scatter(x=df['DateTime'], y=df['ExpOverLoad'], fill='tonexty',name='ExpOverLoad', stackgroup='one',),  row=3, col=1)
        #fig.add_trace(go.Scatter(x=df['DateTime'], y=df['ExpSpill'], fill='tonexty',name='ExpSpill', stackgroup='one',),  row=3, col=1)
        #fig.add_trace(go.Scatter(x=df['DateTime'], y=df['ExpRenew'], fill='tonexty',name='ExpRenew', stackgroup='one',),  row=3, col=1)
        # Act vars
        loadFig = go.Figure()
        #loadFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActOverLoad'], fill='tonexty',name='Actual OverLoad', stackgroup='one',line=dict(color='yellow')))
        loadFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActLossLoad'], fill='tozeroy', name='Actual Loss Load', stackgroup='one',line=dict(color=lossHex)))

        loadFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActSpill'], fill='tonexty',name='Actual Spill', stackgroup='one',line=dict(color='#FF7F50')))
        loadFig.add_trace(go.Scatter(x=df['DateTime'], y=df['ActRenew'], fill='tonexty',name='Actual Renewable', stackgroup='one',line=dict(color=renewableHex)))
        loadFig.add_trace(go.Scatter(x=actuals['DateTime'], y=actuals['total'],name='Actual Wind', line=dict(color='royalblue', width=4, dash='dot')))

        loadFig.add_trace(go.Scatter(x=[daterange[timestep], daterange[timestep]], y=[value['load'][0],4500], name='Time Step', line=dict(color="#2F4F4F")))
        # should work but does not: https://github.com/plotly/plotly.py/issues/3065
        # fig.add_vline(x=timeSteps[1], line_dash="dot", row="all", col="1",
        #       annotation_text="Jan 1, 2018 baseline",
        #       annotation_position="bottom right")

        ## HACK
        #print("ts", timeSteps[1])
        #print(datetime.datetime.strptime("2018-09-24 00:05:00.0", "%Y-%m-%d %H:%M:%S.%f" ))
        #print(datetime.datetime.strptime(timeSteps[1], "%Y-%m-%dT%H:%M:%S.%f").timestamp() * 1000)
        #print(datetime.datetime.strptime("2018-09-24", "%Y-%m-%d").timestamp() * 1000)


        # x = datetime.datetime.strptime(timeSteps[0], "%Y-%m-%dT%H:%M:%S.%f").timestamp() * 1000
        # x1 = datetime.datetime.strptime(timeSteps[5], "%Y-%m-%dT%H:%M:%S.%f").timestamp() * 1000
        # print("x", type(x))
        # fig.add_vrect(x0=x, x1=x1, line_dash="dot", row="all", col=1,
        #               annotation_text="Test",
        #               annotation_position="top right")

        costFig.update_layout(
            autosize=False,
            width=625,
            height=200,
            margin=dict(
                l=50,
                r=25,
                b=50,
                t=20,
                pad=10
            ),
            legend=dict(
                yanchor="bottom",
                y=.99,
                xanchor="right",
                x=0.99
            )
        )
        costFig.update_xaxes(
            #tickangle = 90,
            title_text = "Time",
            showgrid= False,
            range=[daterange[0],daterange[len(daterange)-1]]
            #title_font = {"size": 20},
            #title_standoff = -10
        )
        costFig.update_yaxes(
            #tickangle = 90,
            title_text = "Cost ($/minute)",
            showgrid= False,
            range=[0, 120000],
            #title_font = {"size": 14},
            title_font=dict(size=10),
            title_standoff = 25
        )
        # costFig2.update_layout(
        #     autosize=False,
        #     width=750,
        #     height=300,
        #     margin=dict(
        #         l=50,
        #         r=50,
        #         b=0,
        #         t=0,
        #         pad=0
        #     )
        # )
        loadFig.update_layout(
            autosize=False,
            width=625,
            height=200,
            margin=dict(
                l=55,
                r=25,
                b=50,
                t=20,
                pad=10
            ),
            legend=dict(
                yanchor="bottom",
                y=.99,
                xanchor="right",
                x=0.99
            )
        )
        loadFig.update_xaxes(
            #tickangle = 90,
            title_text = "Time",
            range=[daterange[0],daterange[len(daterange)-1]],
            #title_font = {"size": 20},
            #title_standoff = 25,
            showgrid = False)
        loadFig.update_yaxes(
            #tickangle = 90,
            title_text = "Capacity kW",
            showgrid= False,
            title_font=dict(size=10),
            range=[0, 5000],#[value['load'][0], value['load'][1]+50]
            #title_font = {"size": 20},
            #title_standoff = 25
        )

        #print(daterange[timestep])
        #loadFig.add_vline(x="2020-07-02T00:15:00.0", line_width=3, line_dash="dash",line_color="green")
        return [costFig, loadFig] #[costFig, costFig2, loadFig]
    else:
        raise PreventUpdate()

#------------------------------------------------#
# Callbacks for detail files
#------------------------------------------------#
@app.callback(Output("network-map", "layers"),
              #Output("mapLegend","layers")],
              [Input('timestep', 'data'),
               Input('busFile', 'data'),
               #Input('branchFile', 'data')
               ],
              [#State('busFile', 'data'),
               #State('branchFile', 'data'),
               #State('busVariable', 'data'),
               #State('branchVariable', 'data'),
               #State('busDataColor', 'data'),
               #State('branchDataColor', 'data'),
               State('busLossExtents', 'data'),
               State('busThermalExtents', 'data'),
               State('busRenewableExtents', 'data')],
              prevent_initial_call=True)
def createMap(timestep, busFile, busLossExtents, busThermalExtents, busRenewableExtents):
    #branchFile, busFileState, branchFileState, busVariable, branchVariable, busColor, branchColor,):
    #print("update map",timestep, busVariable, branchFile, busFileState, branchFileState)

    #print(busFile)

    if(timestep == None):
        raise PreventUpdate()

    #print(daterange[timestep])
    # Create the map (and legend) layers
    mapLayers=[]
    legendLayers=[]

    # If we have branch data
    # if(branchFile != None):
    #
    #     # Grab the current timeste of the bus data
    #     branchData = gl.branchDataStore.loc[gl.branchDataStore['DateTime'] == timeSteps[timestep]].values[0].tolist()
    #     branchData.pop(0) # (remove the timestamp)
    #
    #     # Append the data to the geometry data
    #     branchMapData = branch
    #     branchMapData[branchVariable] = branchData
    #
    #     # Get the extent of the data
    #     branchExtents = [min(branchData), max(branchData)]
    #
    #     #colorscales = px.colors.named_colorscales()
    #     #print("colors?" , colorscales)
    #     #plotly.colors.colorscale_to_colors(colorscales[0])
    #     #print(px.colors.sequential.Plasma)
    #
    #     lineLayer = {
    #         "type": "LineLayer",
    #         "id": 'line-layer',
    #         "data": branchMapData.to_dict(orient='records'),
    #         "pickable": False,
    #         "opacity": 0.8,
    #         "widthScale": 1,
    #         "widthMinPixels":1,
    #         "widthMaxPixels":5,
    #         "getWidth": "function(d){return d['"+branchVariable+"']}",
    #         "getSourcePosition": "function(d){return d['fromPos']}",
    #         "getTargetPosition": "function(d){return d['toPos']}" ,
    #         "getColor": {"scale":{"type":colorMaps['branch']['scale'],
    #                               "range":  colorMaps['branch']['rangeRGB'],
    #                               "domain": branchExtents,
    #                               "value": branchVariable,
    #                              }}
    #
    #
    #     }
    #     mapLayers[0] = lineLayer

    # If we have bus data
    if(busFile != None):

        # Grab the current timestep of the bus data, rename column to the variable name
        mapLayers.append(initialLayers[0])
        mapLayers.append(initialLayers[1])

        if(not gl.busLossStore.empty):
            busLoss = gl.busLossStore[['Bus ID',daterange[timestep]]]
            busLoss = busLoss.astype({'Bus ID': 'int64', daterange[timestep]:'float64'})
            busLoss = busLoss.rename(columns={daterange[timestep]: "Loss"})
            #print(busLoss)

        #print("gl", gl.busThermalStore)
        if(not gl.busThermalStore.empty):
            busThermal= gl.busThermalStore[['Bus ID',daterange[timestep]]]
            busThermal = busThermal.astype({'Bus ID': 'int64', daterange[timestep]:'float64'})
            busThermal = busThermal.rename(columns={daterange[timestep]: "Thermal"})

        busRenewable = gl.busRenewableStore[['Bus ID',daterange[timestep]]]
        busRenewable = busRenewable.astype({'Bus ID': 'int64', daterange[timestep]:'float64'})
        busRenewable = busRenewable.rename(columns={daterange[timestep]: "Renewable"})

        # Append the data to the geometry data
        busMapData = pd.merge(busLoss, bus, on='Bus ID')
        busMapData = pd.merge(busThermal, busMapData, on='Bus ID')
        busMapData = pd.merge(busRenewable, busMapData, on='Bus ID')

        # lossColor = {"scale":{"type":colorMaps['whiteRed']['scale'],
        #              "range": colorMaps['whiteRed']['rangeRGB'],
        #              "domain": busLossExtents,
        #              "value": "Loss",
        #             }}
        # thermalColor = {"scale":{"type":colorMaps["blu"]['scale'],
        #                 "range": colorMaps["blu"]['rangeRGB'],
        #                 "domain": busThermalExtents,
        #                 "value": "Thermal",
        #                 }}
        # renewableColor = {"scale":{"type":colorMaps['grn']['scale'],
        #                   "range": colorMaps['grn']['rangeRGB'],
        #                   "domain": busRenewableExtents,
        #                   "value": "Renewable",
        #               }}




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

            #"getRadius": "function(d){return d['Loss']}",
            #"getRadius": "function(d){return d['Thermal']}",
            "getRadius": "function(d){return d['Renewable']}",

            #"getLineWidth": "function(d){return d['Loss']}",
            #"getLineWidth": "function(d){return d['Thermal']}",
            "getLineWidth": "function(d){return d['Renewable']}",

            #"getLineColor": lossColor,
            "getLineColor": renewableColor,
            #"getLineColor": renewableColor,

            #"getFillColor": lossColor,
            #"getFillColor": thermalColor,
            "getFillColor": renewableColor,
        }
        mapLayers.append(renewableScatterLayer)


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

            #"getRadius": "function(d){return d['Loss']}",
            "getRadius": "function(d){return d['Thermal']}",
            #"getRadius": "function(d){return d['Renewable']}",

            #"getLineWidth": "function(d){return d['Loss']}",
            "getLineWidth": "function(d){return d['Thermal']}",
            #"getLineWidth": "function(d){return d['Renewable']}",

            #"getLineColor": lossColor,
            "getLineColor": thermalColor,
            #"getLineColor": renewableColor,

            #"getFillColor": lossColor,
            #"getFillColor": thermalColor,
            "getFillColor": thermalColor,
        }
        mapLayers.append(thermalScatterLayer)


        scatterLayer = {
            "type": "ScatterplotLayer",
            "id": 'scatterplot-layer2',
            "data": busMapData.to_dict(orient='records'),
            "pickable": False,
            "opacity": 0.5,
            "stroked": True,
            "filled": True,
            "radiusScale":50,
            "lineWidthScale": 25,
            "radiusMinPixels": 5,
            "lineWidthMinPixels": 0,
            "getPosition": "function(d){return [d['lng'], d['lat']]}",

            "getRadius": "function(d){return d['Loss']}",
            #"getRadius": "function(d){return d['Thermal']}",
            #"getRadius": "function(d){return d['Renewable']}",

            "getLineWidth": "function(d){return d['Loss']}",
            #"getLineWidth": "function(d){return d['Thermal']}",
            #"getLineWidth": "function(d){return d['Renewable']}",

            #"getLineColor": [0,0,0],
            #"getLineColor": [227,26,28],#lossColor,
            "getLineColor": "function(d){ let color = d['Loss']==0 ? [248, 248, 255] : [227,26,28]; return color}",#lossColor,
            #"getLineColor": thermalColor,
            #"getLineColor": renewableColor,

            #"getFillColor": [227,26,28],
            "getFillColor": "function(d){ let color = d['Loss']==0 ? [248, 248, 255] : [227,26,28]; return color}",

            #if d['Loss'] == 0 return return d['Thermal']}",#lossColor,
            #"getFillColor": thermalColor,
            #"getFillColor": renewableColor
        }
        mapLayers.append(scatterLayer)

        # Set the map legend
        # layers={
        #     'type': "colorLegend",
        #     'id': "dataLayer",
        #     'value': "thermal",
        #     'title': "Thermal",
        #     'position': [0,0],
        #     "colorMap": thermalColor
        # }
        # legendLayers.append(layers);


    # If we haven't generated layers, push the initial layers
    if(len(mapLayers) == 0):
        mapLayers = initialLayers


    return mapLayers#,legendLayers


if __name__ == '__main__':
    app.run_server(debug=True)

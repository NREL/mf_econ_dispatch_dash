import pandas as pd

# ----------------------------------------------------------- #

detailsVars = { 'ActLossLoad':{"label":"Loss of Load"},
                'ActOverLoad':{"label":"Overload"},
                'ActRenew':{"label":"Actual Renewable Generation"},
                'ActSpill':{"label": "Renewable Spill"},
                'ActFirstStageCost':{"label":"First Stage Cost"},
                'ActSecondStageCost':{"label": "Second Stage Cost"}
                }

# Define labels and colors
scenarioDirs = {'details50_AC_72hrs':{"label":"AC OPF", "summary":"ac_results.csv"},
                'details50_DC_72hrs':{"label":"DC OPF", "summary":"dc_results.csv"},
                'details50_CP_72hrs':{"label":"Copper Plate OPF (Real power only)", "summary":"cp_results.csv"},
                'details50_ACCP_72hrs':{"label":"Multi-Fidelity AC-CP OPF", "summary":"mf_ACCP_results.csv"},
                'details50_DCCP_72hrs':{"label":"Multi-Fidelity DC-CP OPF", "summary":"mf_DCCP_results.csv"},
                'details50_cp_ac_conseq':{"label":"Copper Plate with AC consequences (Includes reactive power)", "summary":"cp_ac_actuals_results.csv"},
                'details50_cp_dc_conseq':{"label":"Copper Plate with DC consequences (Real power only)", "summary":"cp_dc_actuals_results.csv"}}
detailsFiles = {'real_loss_of_load.csv':{'label':'Real Loss of Load','color':'whiteRed', 'process':False},
                'real_thermal_set_points_processed.csv':{'label':'Real Thermal Set Points','color':'whiteRed', 'process':True},
                'real_renewable_setpoints_processed.csv':{'label':'Real Renewable Setpoints','color':'whiteRed', 'process':True},
                'real_renewable_spill_processed.csv':{'label':'Real Renewable Spill','color':'whiteRed', 'process':False},
                'real_overload.csv':{'label':'Real Overload','color':'whiteRed', 'process':False},
                #'reactive_overload.csv':{'title':'Reactive Overload','color':'whiteRed', 'process':False},

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
r1Color=[0, 139, 139]
r2Color=[240, 128, 128]
diffColor = [250, 128, 114]
lineColor = [119, 136, 153]

# ----------------------------------------------------------- #
# Time helper functions
def unixTimeMillis(dt):
    ''' Convert datetime to unix timestamp '''
    return int(time.mktime(dt.timetuple()))
def unixToDatetime(unix):
    ''' Convert unix timestamp to datetime. '''
    return pd.to_datetime(unix,unit='s')

# Returns the marks for labeling the slider
def getMarks(timeSteps):
    step = (len(timeSteps)-1)/6
    result = {}
    for i in range(0, len(timeSteps), int(step)):
        result[i] = {'label':pd.to_datetime(timeSteps[i]).strftime("%m/%d,%H:%M")}
                    #'style':

    return result

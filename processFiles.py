# Read in all of the REDUCED scenarios,
# write the private land ownership, save back to csv
# Written by Kristi Potter - 1/17/2020
import pandas as pd
import numpy as np
import pathlib

import os, sys
import json


# Set the path to the data
DATA_PATH = pathlib.Path(__file__).parent.joinpath("./data/").resolve()
SCENARIO_PATH = DATA_PATH.joinpath("50Scen_3days 2")

# Process the network
if True:
    # Read in the network
    bus = pd.read_csv(DATA_PATH.joinpath("RTS/bus.csv"))
    bus = bus[['Bus ID','Bus Name','lat','lng']]

    # Iterate over the bus IDs and add columns
    ids = bus['Bus ID'].to_list()


    scenarioDirs = ['details50_AC_72hrs',
                    'details50_DC_72hrs',
                    'details50_CP_72hrs',
                    'details50_ACCP_72hrs',
                    'details50_DCCP_72hrs',
                    'details50_cp_ac_conseq',
                    'details50_cp_dc_conseq']

    #dataFiles = ['real_thermal_set_points.csv','real_renewable_setpoints.csv',
    #             'reactive_thermal_set_points.csv']
    dataFiles = ['real_renewable_spill.csv']


    for s in scenarioDirs:
        for f in dataFiles:
            print("file", f)
            newFilename = f.split(".")[0]+"_processed."+f.split(".")[1]
            print(newFilename)

            if(os.path.exists(newFilename)):
                continue
            bd = pd.read_csv(SCENARIO_PATH.joinpath(s+"/"+f))


            newBD = {}
            newBD['DateTime'] = bd['DateTime'].to_list()

            # Sum for all the ids
            for i in ids:
                cols = [col for col in bd.columns if str(i) in col]
                temp = bd[cols]
                if(temp.empty):
                    l = [0]*len(bd)
                    newBD[i]=l
                else:
                    newBD[i] = temp.sum(axis=1).to_list()

            # Create a dataframe and transpose
            df = pd.DataFrame.from_dict(newBD, orient='index').T

            print("HERE")
            # Write to file
            print(SCENARIO_PATH.joinpath(s+"/"+newFilename))
            df.to_csv(SCENARIO_PATH.joinpath(s+"/"+newFilename), index=False)
            #break
        #break

# Process the max wind generation
else:
    # Read in the generator info
    gen = pd.read_csv(DATA_PATH.joinpath("RTS/generator_info.csv"))
    gen = gen[['BusNum', 'GenMWMax']]
    gen =gen.groupby(['BusNum']).sum()
    gen.to_csv(DATA_PATH.joinpath("RTS/maxGen.csv"))

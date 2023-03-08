import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from scipy import signal
from scipy.stats import zscore

import sys


from rica.domain.vibration import vibration
from rica.domain.configure import configure
from rica.use_cases.filter import create_filter_bank, filter_signal
from rica.use_cases.power import (compute_min_max_power, compute_power,
                                  create_power_object)
from rica.domain.density import ExtendedKMeans
from rica.use_cases.density import create_densityModule_object, model_predict, load_ExtendedKMeans

def arg_parse():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        dest="FOLDER_LOCATION",
        help="location of the folder containing raw_data"
    )

    parser.add_argument(
        "-cf",
        dest="CALIB_FOLDER",
        help="location of the folder containing calib.yml"
    )

    # parser.add_argument(
    #     "-mf",
    #     dest="MODEL_FOLDER",
    #     help="location of the folder containing model.pkl"
    # )

    parser.add_argument(
        "-sf",
        dest="SAVE_FOLDER",
        help="location of the folder to_save_ the outputs"
    )

    flags=vars(parser.parse_args())
    return flags


def read_params(folder_location):
    with open(folder_location/"params.yml","r") as fid:
        d=yaml.safe_load(fid)
    print(d["fundFreq"])
    return d


def convert_to_vibration(df):
    """Compute vibration objects from dataframe"""
    vibration_list = []
    for i in range(127, 128*(len(df)//128), 127):
        vibration_list.append(
            vibration(
                timestamp=df.loc[i, "time"],
                data=df.loc[i - 127 : i, "vibration"].values,
            )
        )
    print(len(vibration_list))    
    return vibration_list

def compute_all_power(filter_object, power_object, vibration_data):
    # pass vibration data through the power calculator
    filtered_list = [
        filter_signal(filter_object, vibration) for vibration in vibration_data
    ]
    filtered_output_list = [filtered_list[i-1] + filtered_list[i] for i in range(1, len(filtered_list))]
    power_list = [
        compute_power(power_object, filterOutput) for filterOutput in filtered_output_list
    ]
    return power_list


def power_to_df(power_list):
    df = pd.DataFrame([])
    for p in power_list:
        df = df.append(p.to_dict(), ignore_index=True)
    df.timestamp = pd.to_datetime(df.timestamp)
    return df


def get_power(full_vib_list, params):
     # Create filter & Power objects
    oFilter = create_filter_bank(params.copy())
    (oPower, _) = create_power_object(params.copy())

    # Get Raw Power
    power_list = compute_all_power(oFilter, oPower, full_vib_list)
    power_df = power_to_df(power_list)

    # Scale total Power
    myMax = power_df['total_power'].quantile(0.95)     #### Are we still doing this here to get 
    # myMax = power_df['total_power'].max()
    myMin = power_df['total_power'].min()
    power_df['total_power'] = 100 * (power_df['total_power'] - myMin) / (myMax - myMin)
    power_df['total_power'] = power_df['total_power'].clip(lower=0, upper=100)

    return power_df


if __name__=="__main__":
    flags=arg_parse()
    folder_location=Path(flags["FOLDER_LOCATION"])
    calib_folder=Path(flags["CALIB_FOLDER"])
    # model_folder=Path(flags["MODEL_FOLDER"])
    save_folder=Path(flags["SAVE_FOLDER"])

    params=read_params(calib_folder)

    file_names=os.listdir(folder_location)
    file_names=[x for x in file_names if "raw_data" in x]
    print(file_names)
    location_in_time=pd.read_csv(folder_location/"location_in_time.csv")
    location_in_time.to_csv(save_folder/"location_in_time.csv")

    passes=[int(x) for x in location_in_time["passID"].values]
    for i in passes:
        raw_data=pd.read_csv(folder_location/f"raw_data_pass_{i}.csv")
        vib_list=convert_to_vibration(raw_data)
        normalized_power=get_power(vib_list,params)
        normalized_power.to_csv(save_folder/f"normalized_power_pass_{i}.csv")
        print(f"Describe power_df pass {i}:\n{normalized_power.describe()}\n")
        try:
            all_data=pd.concat([all_data,normalized_power])
        except:
            all_data=normalized_power
    all_data.to_csv(save_folder/f"normalized_all_power.csv")


        

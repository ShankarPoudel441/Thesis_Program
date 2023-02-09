import os
import argparse

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

from scipy import signal
from scipy.stats import zscore

from rica.domain.vibration import vibration
from rica.use_cases.filter import create_filter_bank, filter_signal
from rica.use_cases.power import (compute_min_max_power, compute_power,
                                  create_power_object)


def arg_parse():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        dest="FOLDER_LOCATION",
        help="folder location of the data location containing location_in_time.csv"
    )
    flags=vars(parser.parse_args())
    return flags


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


def estimate_fundFreq(df):

    f, _, Sxx = signal.spectrogram(
        df.vibration.values,
        1000,
        "boxcar",
        nperseg=256,
        noverlap=128,
        scaling="density",
        mode="magnitude",
    )

    S_mean = np.mean(Sxx, axis=1)
    fundFreq = f[np.argmax(S_mean)]
    return fundFreq


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


def get_params(fund,full_vib_list):
    params={"fundFreq":float(fund)}
    params["mean"] = [0, 0, 0, 0, 0, 0, 0]
    params["std"] = [1, 1, 1, 1, 1, 1, 1]
    # print("Initial params:\n",params)

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

    print("Describe:\n",power_df.describe(),"\n")

    normalized_power=power_df.copy()
    params["mean"]=list([float(x) for x in power_df.iloc[:,2:].mean().values])
    params["std"]=list([float(x) for x in power_df.iloc[:,2:].std().values])
    
    normalized_power.iloc[:,2:]=normalized_power.iloc[:,2:].apply(zscore)

    print("power_df head\n",power_df.head())
    print(f"params is \n {params}")

    return params, power_df, normalized_power


def save_params(folder_location,params):
    with open(folder_location/ "params.yml", "w") as fid:
        yaml.dump(params, fid, default_flow_style=False)


if __name__=="__main__":
    flags=arg_parse()
    folder_location=Path(flags["FOLDER_LOCATION"])
    file_names=os.listdir(folder_location)
    raw_files=[x for x in file_names if "raw_data_pass" in x]
    print(raw_files)
    
    location_in_time_df=pd.read_csv(folder_location/"location_in_time.csv")
    location_in_time=np.array(location_in_time_df).T[1:].T
    
    all_vib_data_list_of_df = [pd.read_csv(folder_location/x) for x in raw_files]    # List of vibration data in each pass
    full_vib_df=pd.concat(all_vib_data_list_of_df)                                   # Total vibration data in single dataframe
    all_vib_list = [convert_to_vibration(x) for x in all_vib_data_list_of_df]                   # Conversion of each vibration df into list of vibration objects
    full_vib_list = np.concatenate(all_vib_list)                                     # Combining all the lists of vibration object into single list of vibration objects 
    print(len(full_vib_list))

    fund = estimate_fundFreq(full_vib_df)
    print("Fundamental Frequency")
    print(f"\tEstimated: {fund:0.2f}")

    params,power_df,normalized_power =get_params(fund, full_vib_list)

    save_params(folder_location,params)

    power_df.to_csv(folder_location/"non_normalizd_all_power.csv")
    normalized_power.to_csv(folder_location/"normalized_all_power_calibration.csv")


    

    

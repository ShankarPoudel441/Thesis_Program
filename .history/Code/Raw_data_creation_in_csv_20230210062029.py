from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
from scipy import signal

import argparse
from pathlib import Path

def input_args():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        dest="FOLDER_LOCATION",
        help="location of the folder containing location_in_time.csv and to store the raw data"
    )
    
    parser.add_argument(
        "-r"
    )
    
    flags=vars(parser.parse_args())
    return flags



def save_raw(start,stop,passID,to_save_location):
    raw_data=query_betn(start,stop)
    raw_data.to_csv(Path(to_save_location)/f"raw_data_pass_{passID}.csv")

if __name__=="__main__":
    flags=input_args()

    folder_location=flags["FOLDER_LOCATION"]
    location_in_time_df=pd.read_csv(Path(folder_location)/"location_in_time.csv")
    location_in_time=np.array(location_in_time_df).T[1:].T

    print(location_in_time)


    [save_raw(start,stop,passID,folder_location) for start,stop,passID in location_in_time]


    
    

    



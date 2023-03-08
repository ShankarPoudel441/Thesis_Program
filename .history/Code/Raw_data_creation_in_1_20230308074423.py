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
    flags=vars(parser.parse_args())
    return flags

<<<<<<< HEAD
def query_it(str1,client = InfluxDBClient(host="127.0.0.1", port=8086, database="lathrop")):
=======
def query_it(str1,client = InfluxDBClient(host="127.0.0.1", port=8086, database="mchenry_all")):
>>>>>>> e84f0b121b2c5c966c65d807a49be57a76d4a84c
        x=client.query(str1)
        data = pd.DataFrame(x.get_points())
    #     data.time = pd.to_datetime(data.time)
        return data

<<<<<<< HEAD
def query_betn(s_datetime,e_datetime,table="raw",client = InfluxDBClient(host="127.0.0.1", port=8086, database="lathrop")):
    x = client.query(
        f"select * from {table} where time>='{s_datetime}' and time<'{e_datetime}'"
    )
    data=pd.DataFrame(x.get_points())
=======
def query_betn(s_datetime,e_datetime,table="raw",client = InfluxDBClient(host="127.0.0.1", port=8086, database="mchenry_all")):
    query=f"select * from {table} where time>='{s_datetime}' and time<'{e_datetime}'"
    print(query)
    x = client.query(query)
    data=pd.DataFrame(x.get_points())
    print(len(data))
>>>>>>> e84f0b121b2c5c966c65d807a49be57a76d4a84c
    data.time=pd.to_datetime(data.time)
    return data

def save_raw(start,stop,passID,to_save_location):
    raw_data=query_betn(start,stop)
    raw_data.to_csv(Path(to_save_location)/f"raw_data_pass_{passID}.csv")

if __name__=="__main__":
    flags=input_args()

    folder_location=flags["FOLDER_LOCATION"]
    location_in_time_df=pd.read_csv(Path(folder_location)/"location_in_time.csv")
    location_in_time=np.array(location_in_time_df).T[1:].T

    print(location_in_time)


<<<<<<< HEAD
    [save_raw(start,stop,passID,folder_location) for start,stop,passID in location_in_time]
=======
    [save_raw(start,stop,passID,folder_location) for start,stop,passID in location_in_time[:,:3]]
>>>>>>> e84f0b121b2c5c966c65d807a49be57a76d4a84c


    
    

    



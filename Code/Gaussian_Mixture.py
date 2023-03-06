import argparse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import pickle

from pathlib import Path
import os
import yaml

def input_args():
    parser=argparse.ArgumentParser()

    # parser.add_argument(
    #     "-s",
    #     dest="SAVE_AT",
    #     help="folder to save the output dataframes at",
    # )
    parser.add_argument(
        "-f",
        dest="FOLDER_LOCATION",
        help="folder containing the 'normalized_all_power.csv' to train if to train and containing normalized_power_pass_xxx.csv if to predict"
    )
    parser.add_argument(
        "-p",
        dest="PARAMS",
        help="params for the given normalization method"
    )
    parser.add_argument(
        "-m",
        dest="MODEL",
        default=None,
        help="location of the trained model, if none is provided, new model is trained and stoted at -s folder in -mn name"
    )
    parser.add_argument(
        "-mn",
        dest="MODEL_NAME",
        default=None,
        help="Name to save the new created model"
    )
    flags = vars(parser.parse_args())
    return flags

def read_params(folder_location):
    with open(folder_location/"params.yml","r") as fid:
        d=yaml.safe_load(fid)
    print(d["fundFreq"])
    return d

def data_ready_to_fit_test(power_df):
    # reshape Training Data
    x = power_df.to_dict("records")
    x = [[d["p0"], d["p1"], d["p2"], d["p3"], d["p4"], d["p5"], d["p6"]] for d in x]
    np_array_train = np.array(x)
    return np_array_train

def computeTotalPower(params, x):
    """x is a list of seven power coefficients"""
    """Compute total power from powerbands."""
    print(x)
    fundFreq=params["fundFreq"]
    std=params["std"]
    mean=params["mean"]
    f = (
        np.array(
            [
                fundFreq,
                2 * fundFreq,
                3 * fundFreq,
                4 * fundFreq,
                5 * fundFreq,
                6 * fundFreq,
                7 * fundFreq,
            ]
        )
        ** 2
    )

    return np.sum([(a*b+c)*d for a,b,c,d in zip(x,std,mean,f)])/1000000 


def get_the_renaming_of_cluster_by_sorting(model,predictions):
    model_means = model.means_.copy()

    prev_id=range(len(model_means[:,0]))

    ## Using p0 to sort
    # idx = N[:, 0].argsort()

    ## Using total power to sort
    total_power_centers=np.array([computeTotalPower(params,x) for x in model_means])
    idx=total_power_centers.argsort()
    print("total_power_centers",total_power_centers, "idxes", idx)


    conversion={k:v for k,v in zip(prev_id,idx)}
    predictions_n=[conversion[i] for i in predictions]
    return predictions_n






def predict_each_passes_separately(folder_location,model_folder_save,model,params):
    location_in_time=pd.read_csv(folder_location/"location_in_time.csv")
    for i in location_in_time["passID"].values:
        power_df=pd.read_csv(folder_location/f"normalized_power_pass_{i}.csv")
        power_array=data_ready_to_fit_test(power_df)

        prediction=model.predict(power_array)
        f_prediction=get_the_renaming_of_cluster_by_sorting(model,prediction)
        power_df["cluster"]=f_prediction
        power_df.to_csv(folder_location/model_folder_save/f"cluster_predicted_pass_{i}.csv")


if __name__=="__main__":
    flags=input_args()

    folder_location=Path(flags["FOLDER_LOCATION"])
    # save_at=Path(flags["SAVE_AT"])
    model_l=flags["MODEL"]                                     # model_l specifies that model is trained an is to be used from this folder that contains model.pkl
    # save_name=flags["SAVE_NAME"]
    model_name=flags["MODEL_NAME"]
    params = flags["PARAMS"]

    params=read_params(folder_location)

    if model_l:
        with open(folder_location/model_l/"model.pkl", 'rb') as f:
            model = pickle.load(f)
        predict_each_passes_separately(folder_location,model_l,model,params)
    else:
        training_data=pd.read_csv(folder_location/"normalized_all_power.csv")
        np_array_train=data_ready_to_fit_test(training_data)


        model=GaussianMixture(n_components=5,covariance_type='tied',random_state=42)
        model.fit(np_array_train)
        if not os.path.exists(folder_location/model_name):
            os.makedirs(folder_location/model_name)
        with open(folder_location/model_name/"model.pkl",'wb') as f:
            pickle.dump(model,f)

        predict_each_passes_separately(folder_location,model_name,model,params)


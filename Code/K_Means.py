import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle

from pathlib import Path
import os

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
    # parser.add_argument(
    #     "-sn",
    #     dest="SAVE_NAME",
    #     help="name of the predicted dataframe"
    # )
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

# def computeTotalPower(fundFreq, x):
#     """Compute total power from powerbands."""
#     f = (
#         np.array(
#             [
#                 fundFreq,
#                 2 * fundFreq,
#                 3 * fundFreq,
#                 4 * fundFreq,
#                 5 * fundFreq,
#                 6 * fundFreq,
#                 7 * fundFreq,
#             ]
#         )
#         ** 2
#     )
#     return {
#         "total_power": np.matmul(
#             f, 
#         )
#         / 1000000  # Divide by 1 million is from Paloma's experiments
#     }

def sort_clusters_by_total_power(model):
    """Order each cluster by sum"""
    N = model.cluster_centers_.copy()
    sum_N = np.sum(N, axis=1)
    logging.debug(f"1. {sum_N}")
    sorted_N = np.argsort(sum_N)
    logging.debug(f"2. {sorted_N}")
    model.cluster_centers_ = N[sorted_N, :]
    return model

def sort_clusters(model):
    """Sort clusters by power bands."""
    N = model.cluster_centers_.copy()
    for i in range(model.n_features_in_):
        idx = np.argsort(N[:, i])
        idx = N[:, i].argsort()
        model.cluster_centers_[:, i] = N[idx, i]
    print(model.cluster_centers_)
    return model

def sort_cluster_by_po(model):
    """Sort clusters by power bands."""
    N = model.cluster_centers_.copy()
    idx=N.T[0].argsort()
    model.cluster_centers_=N[idx]
    print(model.cluster_centers_)
    return model

def data_ready_to_fit_test(power_df):
    # reshape Training Data
    x = power_df.to_dict("records")
    x = [[d["p0"], d["p1"], d["p2"], d["p3"], d["p4"], d["p5"], d["p6"]] for d in x]
    np_array_train = np.array(x)
    return np_array_train


def predict_each_passes_separately(folder_location,model_folder_save,model):
    location_in_time=pd.read_csv(folder_location/"location_in_time.csv")
    for i in location_in_time["passID"].values:
        power_df=pd.read_csv(folder_location/f"normalized_power_pass_{i}.csv")
        power_array=data_ready_to_fit_test(power_df)

        prediction=model.predict(power_array)
        power_df["cluster"]=prediction
        power_df.to_csv(folder_location/model_folder_save/f"cluster_predicted_pass_{i}.csv")


if __name__=="__main__":
    flags=input_args()

    folder_location=Path(flags["FOLDER_LOCATION"])
    # save_at=Path(flags["SAVE_AT"])
    model_l=flags["MODEL"]                                     # model_l specifies that model is trained an is to be used from this folder that contains model.pkl
    # save_name=flags["SAVE_NAME"]
    model_name=flags["MODEL_NAME"]

    if model_l:
        with open(folder_location/model_l/"model.pkl", 'rb') as f:
            model = pickle.load(f)
        predict_each_passes_separately(folder_location,model_l,model)
    else:
        training_data=pd.read_csv(folder_location/"normalized_all_power.csv")
        np_array_train=data_ready_to_fit_test(training_data)

        model=KMeans(n_clusters=5,random_state=42)
        model.fit(np_array_train)
        model=sort_clusters(model)
        # model=sort_cluster_by_po(model)
        if not os.path.exists(folder_location/model_name):
            os.makedirs(folder_location/model_name)
        with open(folder_location/model_name/"model.pkl",'wb') as f:
            pickle.dump(model,f)

        predict_each_passes_separately(folder_location,model_name,model)


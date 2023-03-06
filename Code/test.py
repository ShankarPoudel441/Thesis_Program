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
        help="location of the trained model"
    )
    parser.add_argument(
        "-sf",
        dest="SAVE_FOLDER",
        default=None,
        help="Name to folder to save in"
    )
    flags = vars(parser.parse_args())
    return flags


def data_ready_to_fit_test(power_df):
    # reshape Training Data
    x = power_df.to_dict("records")
    x = [[d["p0"], d["p1"], d["p2"], d["p3"], d["p4"], d["p5"], d["p6"]] for d in x]
    np_array_train = np.array(x)
    return np_array_train


def predict_each_passes_separately(folder_location,save_folder_save,model):
    location_in_time=pd.read_csv(folder_location/"location_in_time.csv")
    for i in location_in_time["passID"].values:
        power_df=pd.read_csv(folder_location/f"normalized_power_pass_{i}.csv")
        power_array=data_ready_to_fit_test(power_df)

        prediction=model.predict(power_array)
        power_df["cluster"]=prediction
        power_df.to_csv(save_folder/f"cluster_predicted_pass_{i}.csv")


if __name__=="__main__":
    flags=input_args()

    folder_location=Path(flags["FOLDER_LOCATION"])
    model_l=Path(flags["MODEL"] )                                    # model_l specifies that model is trained an is to be used from this folder that contains model.pkl
    save_folder=Path(flags["SAVE_FOLDER"])

    with open(model_l/"model.pkl", 'rb') as f:
        model = pickle.load(f)
    predict_each_passes_separately(folder_location,save_folder,model)

    
    
    
    # else:
    #     training_data=pd.read_csv(folder_location/"normalized_all_power.csv")
    #     np_array_train=data_ready_to_fit_test(training_data)

    #     model=KMeans(n_clusters=5,random_state=42)
    #     model.fit(np_array_train)
    #     model=sort_clusters(model)
    #     # model=sort_cluster_by_po(model)
    #     if not os.path.exists(folder_location/model_name):
    #         os.makedirs(folder_location/model_name)
    #     with open(folder_location/model_name/"model.pkl",'wb') as f:
    #         pickle.dump(model,f)

    #     predict_each_passes_separately(folder_location,model_name,model)


#!/bin/bash
conda activate rica
python Raw_data_creation_in_1.py -f "/home/spoudel/All_Thesis/Thesis_program/Data/McHenry/location_6/" 
python create_config_param.py -f "/home/spoudel/All_Thesis/Thesis_program/Data/McHenry/location_6/"
python create_power.py -f "/home/spoudel/All_Thesis/Thesis_program/Data/McHenry/location_6/"
python K_Means.py -f "/home/spoudel/All_Thesis/Thesis_program/Data/McHenry/location_6/" -mn "Kmeans"
python Gaussian_Mixture.py -f "/home/spoudel/All_Thesis/Thesis_program/Data/McHenry/location_6/" -mn "GMM"

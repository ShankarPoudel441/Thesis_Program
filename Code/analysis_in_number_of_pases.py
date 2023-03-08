import numpy as np
import pandas as pd

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


from time import time
import numpy as np
import matplotlib.path as mpltPath

import math


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
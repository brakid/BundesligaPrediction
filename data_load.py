import prediction

import numpy as np
import pandas as pd
import json
import urllib
from os import listdir, remove
from os.path import isfile, join, exists

if (exists(prediction.teams_file)):
    remove(prediction.teams_file)
if (exists(prediction.matches_file)):
    remove(prediction.matches_file)

encoder = prediction.get_encoder(prediction.get_all_teams()["TeamName"].values)
loaded_data = prediction.prepare_data(encoder)

Y = loaded_data[["0_GameResult", "1_GameResult", "2_GameResult"]]
X = loaded_data.drop(["0_GameResult", "1_GameResult", "2_GameResult"], axis = 1)

X_data = X.values
Y_data = Y.values

prediction.train_model(X_data, Y_data)
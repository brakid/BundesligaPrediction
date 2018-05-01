import numpy as np
import pandas as pd
import json
import urllib
from os import listdir
from os.path import isfile, join, exists
from sklearn.preprocessing import LabelBinarizer
import mxnet as mx
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

np.random.seed(0)
mx.random.seed(0)

BATCH_SIZE = 200

years = xrange(2010, 2018)
table_data_directory = "./data/table"
match_data_directory = "./data/matchdata"
teams_file = "./data/teams.csv"
matches_file = "./data/matches.csv"

def get_team_name(team, teams):
    if (team not in teams):
        return "unknown"
    else:
        return team

def get_encoder(teams):
    encoder = LabelBinarizer()
    encoder.fit(teams.tolist() + ["unknown"])
    return encoder
    
def encode_team(team, encoder):
    team_name = get_team_name(team, encoder.classes_)
    return encoder.transform([team_name])[0]

def decode_team(team_vector, encoder):
    return encoder.inverse_transform(team_vector)[0]
    
def get_table(year, from_file=True):
    table_result = None
    if (from_file == True):
        table_result = pd.read_json(table_data_directory + "/table" + str(year) + ".txt")
    else:
        url = "https://www.openligadb.de/api/getbltable/bl1/2017"
        response = urllib.urlopen(url)
        data = json.loads(response.read())
        table_result = pd.DataFrame.from_dict(data)
    table_result = table_result[["TeamName"]]
    table_result["Placement"] = table_result.index + 1
    return table_result

def get_team_placement(team, year, from_file=True):
    table = get_table(year, from_file)
    team_placement = table.loc[table["TeamName"] == team.decode('utf-8')]
    if team_placement.empty:
        return 100 # team was not in BL1
    return team_placement["Placement"].values[0]

def compare_placements(placement_team_1, placement_team_2):
    if placement_team_1 < placement_team_2:
        return np.array([1, 0, 0])
    elif placement_team_1 == placement_team_2:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])
    
def get_all_teams():
    if (exists(teams_file)):
        return pd.read_csv(teams_file, index_col = 0)
    else:
        teams_by_year = []
        for year in years:
            teams_by_year.append(get_table(year)["TeamName"].tolist())

        teams_by_year = np.array(teams_by_year).flatten()
        teams = pd.DataFrame(data = teams_by_year, columns = ["TeamName"]).drop_duplicates().reset_index(drop=True)
        teams.to_csv(teams_file, encoding='utf-8')
        return pd.read_csv(teams_file, index_col = 0)
    
def flatten_data_frame(data_frame, column):
    new_data_frame = pd.DataFrame(data_frame[column].values.tolist())
    new_data_frame.columns = new_data_frame.columns.map(lambda x: str(x) + "_" + str(column))
    return data_frame.join(new_data_frame)

def get_team_1(match_json):
    return match_json["Team1"]["TeamName"]

def get_team_2(match_json):
    return match_json["Team2"]["TeamName"]

def get_goals(match_json):
    goals_frame = pd.DataFrame.from_dict(match_json["MatchResults"])
    
    goals_frame = goals_frame.loc[goals_frame["ResultName"] == "Endergebnis"]
    
    return goals_frame[["PointsTeam1", "PointsTeam2"]].values[0]

def get_goals_team_1(match_json):
    goals = get_goals(match_json)
    
    return goals[0]

def get_goals_team_2(match_json):
    goals = get_goals(match_json)
    
    return goals[1]

def get_match_data(match_json):
    return [get_team_1(match_json), get_team_2(match_json), get_goals_team_1(match_json), get_goals_team_2(match_json)]

def parse_match_data(year):
    content = open(match_data_directory + "/matchdata" + str(2010) + ".txt", "r").read()
    content_json = json.loads(content)
    results = []
    for match_json in content_json:
        results.append(get_match_data(match_json) + [year])
        
    return pd.DataFrame(data = np.array(results), columns = ["Team1", "Team2", "GoalsTeam1", "GoalsTeam2", "Year"])

def parse_matches():
    match_data = []
    for year in years:
        match_data.append(parse_match_data(year))
        
    return pd.concat(match_data).reset_index(drop=True)
  
def compare_goals(goals_team_1, goals_team_2):
    if goals_team_1 > goals_team_2:
        return np.array([1, 0, 0])
    elif goals_team_1 == goals_team_2:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])

def prepare_data(encoder):
    if (exists(matches_file)):
        return pd.read_csv(matches_file, index_col = 0) 
    else:
        m = parse_matches()

        m["Team1Id"] = [encode_team(team_name.encode('utf-8'), encoder) for team_name in m["Team1"]]
        m["Team1Placement"] = [get_team_placement(team_data[0].encode('utf-8'), team_data[1]) for team_data in m[["Team1", "Year"]].values]
        m["Team2Id"] = [encode_team(team_name.encode('utf-8'), encoder) for team_name in m["Team2"]]
        m["Team2Placement"] = [get_team_placement(team_data[0].encode('utf-8'), team_data[1]) for team_data in m[["Team2", "Year"]].values]
        m["GameResult"] = [compare_goals(goals[0], goals[1]) for goals in m[["GoalsTeam1", "GoalsTeam2"]].values]
        m["Placements"] = [compare_placements(placements[0], placements[1]) for placements in m[["Team1Placement", "Team2Placement"]].values]

        m = flatten_data_frame(m, "Team1Id")
        m = flatten_data_frame(m, "Team2Id")
        m = flatten_data_frame(m, "Placements")
        m = flatten_data_frame(m, "GameResult")

        matches = m.drop(["Team1", "Team2", "GoalsTeam1", "GoalsTeam2", "Year", "Team1Placement", "Team2Placement", "Team1Id", "Team2Id", "Placements", "GameResult"], axis = 1)

        matches.to_csv(matches_file, encoding='utf-8')
        return pd.read_csv(matches_file, index_col = 0) 

def train_model(X, Y, evaluation=0.1):
    assert len(X) == len(Y)
    permutation = np.random.permutation(len(X))
    
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    
    evaluation_index = int(round(0.1 * len(X)))
    
    X_train = X_shuffled[evaluation_index:]
    Y_train = Y_shuffled[evaluation_index:]
    X_test = X_shuffled[:evaluation_index]
    Y_test = Y_shuffled[:evaluation_index]
    
    train_iter = mx.io.NDArrayIter(data=X_train, label=Y_train, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = mx.io.NDArrayIter(data=X_test, label=Y_test, batch_size=BATCH_SIZE, shuffle=True)
    
    data = mx.symbol.Variable("data")
    fc1 = mx.symbol.FullyConnected(data, name="fc1", num_hidden=59)
    act1 = mx.symbol.Activation(fc1, name="relu1", act_type="sigmoid")
    fc2 = mx.symbol.FullyConnected(act1, name="fc2", num_hidden=128)
    act2 = mx.symbol.Activation(fc2, name="relu2", act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name="fc3", num_hidden=3)
    softmax = mx.symbol.SoftmaxOutput(fc3, name="softmax") # sets loss function as cross-entropy loss

    context = mx.cpu()
    
    model = mx.mod.Module(symbol=softmax, context=context)
    model.fit(train_iter,
              eval_data=test_iter,
              optimizer='adam',
              optimizer_params={'learning_rate':0.01},
              eval_metric='acc',
              num_epoch=2000)
    model.save_checkpoint("bundesliga_model", 0)
        
    return model

def load_model():
    context = mx.cpu()
    try:
        module = mx.mod.Module.load("bundesliga_model", 0)
        module.bind(data_shapes=[('data', (1, 59))], label_shapes = module._label_shapes)
        return module
    except:
        return None

def predict_game(model, data):
    test_iter = mx.io.NDArrayIter(data=np.array([data]), label=None, batch_size=1)
    prob = model.predict(test_iter, num_batch=1)[0].asnumpy()
    return prob

def get_game_prediction(model, encoder, team_name_1, team_name_2):
    team_1_id = encode_team(team_name_1, encoder)
    team_2_id = encode_team(team_name_2, encoder)
    print team_1_id
    print team_2_id
    team_1_placement = get_team_placement(team_name_1, 2017, False)
    team_2_placement = get_team_placement(team_name_2, 2017, False)
    placement_comparison = compare_placements(team_1_placement, team_2_placement)
    print placement_comparison
    team_ids = np.append(team_1_id, team_1_id)
    data = np.append(team_ids, placement_comparison)
    return predict_game(model, data)

#encoder = get_encoder(get_all_teams()["TeamName"].values)
#loaded_data = prepare_data()
#Y = loaded_data[["0_GameResult", "1_GameResult", "2_GameResult"]]
#X = loaded_data.drop(["0_GameResult", "1_GameResult", "2_GameResult"], axis = 1)

#X_data = X.values
#Y_data = Y.values

#data_iter = mx.io.NDArrayIter(data=X_data, label=Y_data, batch_size=BATCH_SIZE, shuffle=True)
#X = X_data[0]
#Y = Y_data[0]

#print get_game_prediction(model, encoder, "FC Bayern", "VfL Wolfsburg")
#print get_game_prediction(model, encoder, "TSG 1899 Hoffenheim", "FC Bayern")
#print get_game_prediction(model, encoder, "FC Bayern", "TSG 1899 Hoffenheim")
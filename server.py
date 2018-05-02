#!flask/bin/python
from flask import Flask, jsonify, abort, request, make_response, url_for, redirect
import datetime
import prediction

def validate_team(team, teams):
    return (team in teams)

encoder = prediction.get_encoder(prediction.get_all_teams()["TeamName"].values)
model = prediction.load_model()

app = Flask(__name__, static_url_path='')

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

@app.route('/')
def get_index():
    return redirect(url_for('static', filename='index.html'))

@app.route('/api/v1/prediction', methods = ['POST'])
def get_prediction():
    print request.json
    if not request.json or (not 'home' in request.json or not 'away' in request.json):
        abort(400)

    home = unicode(request.json['home']).encode('utf-8')
    away = unicode(request.json['away']).encode('utf-8')

    if not validate_team(home, encoder.classes_):
        return make_response(jsonify( { 'error': 'Home team not supported' } ), 400)
    if not validate_team(away, encoder.classes_):
        return make_response(jsonify( { 'error': 'Away team not supported' } ), 400)
    
    game_prediction = prediction.get_game_prediction(model, encoder, home, away)

    return jsonify({ 'homeTeamWins': round(game_prediction[0], 5), 'draw': round(game_prediction[1], 5), 'awayTeamWins': round(game_prediction[2], 5) }), 200

@app.route('/api/v1/teams', defaults={'year': datetime.datetime.now().year - 1})
@app.route('/api/v1/teams/<year>')
def get_teams(year):
    if year < 2010 and year > datetime.datetime.now().year - 1:
        return make_response(jsonify( { 'error': 'Year not supported must be between 2010 and ' + str(datetime.datetime.now().year - 1) } ), 403)

    teams = prediction.get_teams(year)[["TeamName", "TeamIconUrl"]].values.tolist()

    return jsonify([{'teamName': team[0], 'teamIconUrl': team[1]} for team in teams]), 200
    
if __name__ == '__main__':
    app.run(debug = True, port = 8080)
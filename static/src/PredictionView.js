import React from 'react';

export default class PredictionView extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      teams: this.props.teams,
      homeTeam: this.props.homeTeam,
      awayTeam: this.props.awayTeam,
      predictionLoaded: false,
      predictionLoading: false,
      prediction: {}
    };

    this.getTeam = this.getTeam.bind(this);
    this.predictResult = this.predictResult.bind(this);
    this.displayPrediction = this.displayPrediction.bind(this);
  }

  componentWillReceiveProps(nextProps) {
    this.setState({
      teams: nextProps.teams,
      homeTeam: nextProps.homeTeam,
      awayTeam: nextProps.awayTeam,
      predictionLoaded: false,
      predictionLoading: false,
      prediction: {}
    });

    console.log('Received new props: ' + JSON.stringify(nextProps));
  }

  getTeam(teamName) {
    for(var index = 0; index < this.state.teams.length; index++) {
      var team = this.state.teams[index];
      console.log(team.teamName);
      console.log(teamName);
      if (team.teamName === teamName) {
        console.log('Found team: ' + JSON.stringify(team));
        return team;
      }
    }

    console.log('Nothing found');
  }

  predictResult() {
    this.setState({
      teams: this.state.teams,
      homeTeam: this.state.homeTeam,
      awayTeam: this.state.awayTeam,
      predictionLoaded: false,
      predictionLoading: true,
      prediction: {}
    });

    console.log('Retrieving prediction');

    var currentUrl = window.location.href.replace('#', '').replace('/index.html', '');
        
    fetch(currentUrl + '/api/v1/prediction', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({'away':this.state.homeTeam,'home':this.state.awayTeam})
      })
      .then(res => res.json())
      .then((result) => {
        this.setState({
          teams: this.state.teams,
          homeTeam: this.state.homeTeam,
          awayTeam: this.state.awayTeam,
          predictionLoaded: true,
          predictionLoading: true,
          prediction: result
        });
        console.log(JSON.stringify(this.state.prediction));
      })
      .catch((error) => {
        alert(error);
      });
  }

  getPercentage(probability) {
    return (probability * 100).toFixed(2);
  }

  displayPrediction() {
    var homeWinsProbability = this.state.prediction.homeTeamWins;
    var awayWinsProbability = this.state.prediction.awayTeamWins;
    var drawProbability = this.state.prediction.draw;

    if (homeWinsProbability > awayWinsProbability && homeWinsProbability > drawProbability) {
      return this.state.homeTeam + ' gewinnt zuhause gegen ' + this.state.awayTeam + ' (' + this.getPercentage(homeWinsProbability) + '% Konfidenz)';
    } else if (awayWinsProbability > homeWinsProbability && awayWinsProbability > drawProbability) {
      return this.state.awayTeam + ' gewinnt auswärts gegen ' + this.state.homeTeam + ' (' + this.getPercentage(awayWinsProbability) + '% Konfidenz)';
    } else {
      return 'Das Spiel zwischen ' + this.state.awayTeam + ' und ' + this.state.homeTeam + ' endet unentschieden (' + this.getPercentage(drawProbability) + '% Konfidenz)';
    }
  }

  render() {
    var homeTeam = this.getTeam(this.state.homeTeam);
    var awayTeam = this.getTeam(this.state.awayTeam);

    return (
      <div className='card-deck mb-3 text-center'>
        <div className='card mb-6 box-shadow'>
          <div className='card-body'>
            <h1 className='card-title pricing-card-title'><img className='team-icon-medium' src={ homeTeam.teamIconUrl } />{ homeTeam.teamName } : <img className='team-icon-medium' src={ awayTeam.teamIconUrl } />{ awayTeam.teamName }</h1>
            {
              this.state.predictionLoading == false 
                ? (<button type='button' className='btn btn-lg btn-block btn-outline-primary' onClick={ e => this.predictResult() }>Spielergebnis vorhersagen</button>)
                : (<h5 className='card-title'>{
                    this.state.predictionLoaded == false
                      ? (<span>Loading...</span>)
                      : (<span>{ this.displayPrediction() }</span>)
                  }</h5>)
            }
          </div>
        </div>
      </div>
    );
  }
}
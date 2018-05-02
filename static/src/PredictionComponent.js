import React from 'react';

import TeamsSelector from './TeamsSelector';
import PredictionView from './PredictionView';

export default class PredictionComponent extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      teamsLoaded: false,
      teams: [],
      teamsSelected: false,
      homeTeam: '',
      awayTeam: ''
    };

    this.selectedTeams = this.selectedTeams.bind(this);
  }

  componentDidMount() {
    var currentUrl = window.location.href.replace('#', '').replace('/index.html', '');
        
    fetch(currentUrl + '/api/v1/teams')
      .then(res => res.json())
      .then((result) => {
        this.setState({
          teams: result,
          teamsLoaded: true,
          teamsSelected: false,
          homeTeam: '',
          awayTeam: ''
        });
        console.log(JSON.stringify(this.state.teams));
      })
      .catch((error) => {
        alert(error);
      });
  }

  selectedTeams(homeTeam, awayTeam) {
    this.setState({
      teams: this.state.teams,
      teamsLoaded: this.state.teamsLoaded,
      teamsSelected: true,
      homeTeam: homeTeam,
      awayTeam: awayTeam
    });

    console.log('PredictionComponent: ' + homeTeam + ' : ' + awayTeam);
  }

  render() {
    if (this.state.teamsLoaded) {
      return (
        <div>
          {
            this.state.teamsSelected ? (<PredictionView teams={ this.state.teams } homeTeam={ this.state.homeTeam } awayTeam={ this.state.awayTeam }/>) : null
          }
          <TeamsSelector teams={ this.state.teams } selectTeamsHandler={ this.selectedTeams }/>
        </div>
      );
    } else {
      return (
        <p>Loading...</p>
      );
    }
  }
}
import React from 'react';

import TeamSelector from './TeamSelector';

export default class TeamsSelector extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      teams: props.teams,
      homeTeam: '',
      awayTeam: ''
    };

    this.selectHomeTeam = this.selectHomeTeam.bind(this);
    this.selectAwayTeam = this.selectAwayTeam.bind(this);
    this.propagateSelectedTeams = this.propagateSelectedTeams.bind(this);
  }

  propagateSelectedTeams(homeTeam, awayTeam) {
    var teamNames = this.state.teams.map(function(team, index) {
      return team.teamName
    });
    if (teamNames.indexOf(homeTeam) >= 0 && teamNames.indexOf(awayTeam) >= 0) {
      console.log('TeamsSelector: ' + homeTeam + ' : ' + awayTeam);
      this.props.selectTeamsHandler(homeTeam, awayTeam);
    }
  }

  selectHomeTeam(teamName) {
    this.setState({
      teams: this.state.teams,
      homeTeam: teamName,
      awayTeam: this.state.awayTeam
    });

    this.propagateSelectedTeams(teamName, this.state.awayTeam);
  }

  selectAwayTeam(teamName) {
    this.setState({
      teams: this.state.teams,
      homeTeam: this.state.homeTeam,
      awayTeam: teamName
    });

    this.propagateSelectedTeams(this.state.homeTeam, teamName);
  }

  render() {
    return (
      <div className='card-deck mb-3 text-center'>
        <TeamSelector teams={ this.state.teams } caption='Heimteam' selectTeamHander= { this.selectHomeTeam }/>
        <TeamSelector teams={ this.state.teams } caption='Auswärtsteam' selectTeamHander= { this.selectAwayTeam }/>
      </div>
    );
  }
}
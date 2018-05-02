import React from 'react';

export default class TeamSelector extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      teams: props.teams,
      selectedTeam: -1
    };

    this.selectTeam = this.selectTeam.bind(this);
  }

  selectTeam(index) {
    if (index >= 0 && index < this.state.teams.length) {
      var teamName = this.state.teams[index].teamName;
      console.log('Selected: ' + teamName);
      this.setState({
        teams: this.state.teams,
        selectedTeam: index
      });
      this.props.selectTeamHander(teamName);
    }
  }

  render() {
    var selectedTeam = this.state.selectedTeam;
    var selectTeam = this.selectTeam;
    return (
      <div className='card mb-6 box-shadow'>
        <div className='card-header'>
          <h4 className='my-0 font-weight-normal'>{ this.props.caption }</h4>
        </div>
        <div className='card-body'>
          <div className='list-group'>
            {
              this.state.teams.map(function(team, index) {
                return (
                  <a href='#' className={ 'list-group-item list-group-item-action' + (index == selectedTeam ? ' active' : '') } onClick={ e => selectTeam(index) } key={ index }>
                    <img className='team-icon-small' src={ team.teamIconUrl } />{ team.teamName }
                  </a>
                );
              })
            }
          </div>
        </div>
      </div>
    );
  }
}
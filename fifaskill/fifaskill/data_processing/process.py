import numpy as np
import pandas as pd


def win_loss_matrix(data):
    homes = pd.unique(data.home_team)
    aways = pd.unique(data.away_team)
    teams = np.union1d(homes, aways)
    num_teams = np.size(teams)
    team_num_map = dict(zip(teams, range(num_teams)))

    goal_differences = np.zeros((num_teams, num_teams))
    matchups = np.zeros((num_teams, num_teams))
    draws = np.zeros((num_teams, num_teams))

    for row in data.itertuples():
        home = team_num_map[row.home_team]
        away = team_num_map[row.away_team]

        dif = row.home_team_goal - row.away_team_goal
        goal_differences[home, away] += dif
        goal_differences[away, home] -= dif

        if dif > 0:
            matchups[home, away] += 1
            matchups[away, home] -= 1
        elif dif < 0:
            matchups[home, away] -= 1
            matchups[away, home] += 1
        else:
            draws[home, away] += 1
            draws[away, home] += 1

    return matchups, draws, goal_differences, team_num_map


def team_stats(data, team_num_map=None):
    homes = pd.unique(data.home_team)
    aways = pd.unique(data.away_team)
    teams = np.union1d(homes, aways)
    num_teams = np.size(teams)
    if not team_num_map:
        team_num_map = dict(zip(teams, range(num_teams)))

    stats = np.zeros((num_teams, 4))  # win loss draw dif

    for row in data.itertuples():
        home = team_num_map[row.home_team]
        away = team_num_map[row.away_team]

        dif = row.home_team_goal - row.away_team_goal

        if dif > 0:
            stats[home, 0] += 1
            stats[away, 1] += 1
        elif dif < 0:
            stats[home, 1] += 1
            stats[away, 0] += 1
        else:
            stats[home, 2] += 1
            stats[away, 2] += 1

        stats[home, 3] += dif
        stats[away, 3] -= dif

    return stats, team_num_map


def data_stats(data):
    homes = pd.unique(data.home_team)
    aways = pd.unique(data.away_team)
    teams = np.union1d(homes, aways)
    num_teams = np.size(teams)

    num_matches = 0
    num_draws = 0
    num_home_wins = 0
    num_away_wins = 0
    for row in data.itertuples():
        num_matches += 1

        dif = row.home_team_goal - row.away_team_goal

        if dif > 0:
            num_home_wins += 1
        elif dif < 0:
            num_away_wins += 1
        else:
            num_draws += 1

    return num_matches, num_home_wins, num_away_wins, num_draws, num_teams


def partition_data(data, ratio=0.1):
    train = data.shuffle()
    test = train
    return train, test


def rankings(data):
    stats, team_num_map = team_stats(data)

    # inverse map the team map for indices
    team_num_map = {v: k for k, v in team_num_map.items()}

    points = 3*stats[:, 0] + 1*stats[:, 2]

    indices = np.arange(len(team_num_map.keys()))
    data_dict = {'points': pd.Series(points, index=[team_num_map[x] 
                                                    for x in indices]),
                 'goal_dif': pd.Series(stats[:, 3], index=[team_num_map[x]
                                                    for x in indices])
                 }
    points_table = pd.DataFrame(data_dict)

    points_table = points_table.sort_values(by=['points', 'goal_dif'],
                                            ascending=False)

    return points_table

def rankings_from_skills(team_skills, team_num_map, skill_names=['skill']):
    # inverse map the team map for indices
    team_num_map = {v: k for k, v in team_num_map.items()}
    indices = np.arange(len(team_num_map.keys()))

    # to be modified to take in multiple skill names
    data_dict = {'skill': pd.Series(team_skills, index=[team_num_map[x] 
                                                    for x in indices])
                 }
    skills_table = pd.DataFrame(data_dict)
    skills_table = skills_table.sort_values(by=['skill'],
                                            ascending=False)
    return skills_table

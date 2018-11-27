from trueskill import Rating, rate_1vs1, global_env
import trueskill
import itertools
import math
import numpy as np
import pandas as pd
from fifaskill.data_processing import process


class TrueskillModel(object):
    def __init__(self, match_data, score_weighting=False):
        if match_data is None:
            raise ValueError("Data cannot be null")

        self._trained = False

        _, _, team_num_map = process.match_vectors(match_data)

        self.team_num_map = team_num_map

        homes = pd.unique(match_data.home_team)
        aways = pd.unique(match_data.away_team)
        teams = np.union1d(homes, aways)

        self.team_ratings = {}
        for team in teams:
            self.team_ratings[team] = Rating()

        self.avg_margin = 0
        self.num_draws = 0
        trueskill.setup(mu=25.0, sigma=8.333333333333334, beta=4.1666666666666,
                        tau=0.08333333333333334, draw_probability=0.26)
        self.train(match_data, score_weighting)

        self.team_skills = np.zeros((len(teams),))
        for team in teams:
            self.team_skills[self.team_num_map[team]] = \
                             self.team_ratings[team].mu
        return

    def train(self, data, score_weighting=False):
        '''
        Given new match data can add onto team_ratings Data is assumed to be a
        pandas dataframe Score_weighting to be implemented to take into account
        goal diff
        '''
        avg_margin = self.avg_margin * self.num_draws
        num_draws = self.num_draws
        for row in data.itertuples():
            draw = False
            if row.home_team_goal > row.away_team_goal:
                team_win = row.home_team
                team_lose = row.away_team
                # win_goals = row.home_team_goal
                # lose_goals = row.away_team_goal
            elif row.home_team_goal < row.away_team_goal:
                team_win = row.away_team
                team_lose = row.home_team
                # win_goals = row.away_team_goal
                # lose_goals = row.home_team_goal
            else:
                team_win = row.home_team
                team_lose = row.away_team
                # win_goals = row.home_team_goal
                # lose_goals = row.away_team_goal
                draw = True

            rating_win = self.team_ratings[team_win]
            rating_lose = self.team_ratings[team_lose]
            if draw:
                num_draws += 1
                avg_margin += np.abs(rating_win.mu - rating_lose.mu)
            rating_win, rating_lose = rate_1vs1(rating_win, rating_lose,
                                                drawn=draw)
            self.team_ratings[team_win] = rating_win
            self.team_ratings[team_lose] = rating_lose

        self.num_draws = num_draws
        self.avg_margin = avg_margin / self.num_draws
        return

    def _win_probability(self, team1, team2):
        '''
        Adapted from code from Juho Snellman
        https://github.com/sublee/trueskill/issues/1#issuecomment-149762508
        '''
        delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
        sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
        size = len(team1) + len(team2)
        ts = global_env()
        denom = math.sqrt(size * (ts.beta * ts.beta) + sum_sigma)

        return ts.cdf(delta_mu / denom)

    def predict(self, team1, team2, goal_dif=False):
        rating1 = self.team_ratings[team1]
        rating2 = self.team_ratings[team2]

        if rating1.mu - rating2.mu > self.avg_margin:
            return 1
        elif rating2.mu - rating1.mu > self.avg_margin:
            return -1
        else:
            return 0

    def predict_many(self, matchups, goal_dif=False):
        outputs = []
        for matchup in matchups:
            outputs.append(self.predict(matchup[0], matchup[1]))

        return outputs

    def sample(self, team1, team2):
        rating1 = self.team_ratings[team1]
        rating2 = self.team_ratings[team2]

        performance1 = np.random.normal(loc=rating1.mu, scale=rating1.sigma)
        performance2 = np.random.normal(loc=rating2.mu, scale=rating2.sigma)

        if performance1 - performance2 > self.avg_margin:
            return 1
        elif performance2 - performance1 > self.avg_margin:
            return -1
        else:
            return 0

        return 

    def simulate(self, data, num_simulations = 5, prior_loc=0, prior_scale=1):
        results = []
        for row in data.itertuples():
            sims = np.zeros(3)
            for i in range(num_simulations):
                res = self.sample(row.home_team, row.away_team)
                sims[res+1] += 1
            
            results.append(np.argmax(sims)-1)
            
        return results

from trueskill import Rating, quality_1vs1, rate_1vs1, global_env, Trueskill

class TrueskillModel(object):
    def __init__(self, match_data, score_weighting=False):
		homes = pd.unique(match_data.home_team)
		aways = pd.unique(match_data.away_team)
		teams = np.union1d(homes,aways)

		self.team_ratings = {}
		for team in teams:
    		team_ratings[team] = Rating()

    	self.avg_margin = 0
    	self.num_draws = 0
    	self.train(match_data, score_weighting)


    def train(self, data, score_weighting=False):
    	'''
    	Given new match data can add onto team_ratings
    	Data is assumed to be a pandas dataframe
    	Score_weighting to be implemented to take into account goal diff
    	'''
    	avg_margin = self.avg_margin * self_num_draws
    	num_draws = self.num_draws
    	for row in data.itertuples():
		    draw = False
		    if row.home_team_goal > row.away_team_goal:
		        team_win = row.home_team
		        team_lose = row.away_team
		        win_goals = row.home_team_goal
		        lose_goals = row.away_team_goal
		    elif row.home_team_goal < row.away_team_goal:
		        team_win = row.away_team
		        team_lose = row.home_team
		        win_goals = row.away_team_goal
		        lose_goals = row.home_team_goal
		    else:
		        team_win = row.home_team
		        team_lose = row.away_team
		        win_goals = row.home_team_goal
		        lose_goals = row.away_team_goal
		        draw=True
		    
		    rating_win = self.team_ratings[team_win]
		    rating_lose = self.team_ratings[team_lose]
		    if draw:
		    	num_draws += 1
		    	avg_margin += np.abs(rating_win - rating_lose)
		    rating_win, rating_lose = rate_1vs1(rating_win, rating_lose, drawn=draw)
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

	def predict(self, team1, team2):
		if 

	
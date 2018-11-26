import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import Gamma, Poisson, PointMass, Uniform, Dirichlet, Empirical, Categorical, Normal
from fifaskill.data_processing import process

class Toy(object):
    def __init__(self, data = None, goal_dif=False):
        if data is None:
            raise ValueError("Data cannot be null")

        self._trained = False         
        _, _, goal_differences, team_num_map = process.win_loss_matrix(data)
        self.team_num_map = team_num_map
        self.data = goal_differences

        self.train()

    def train(self, n_iter=1000):
        # Model based on Alek's toy model in Jupyter
        n = len(self.team_num_map.keys())

        initial_loc = tf.ones((n,1), dtype='float32') * 25
        initial_scale = tf.ones((n,1),  dtype='float32') * (25/3)**2

        team_skill = Normal(loc=initial_loc, scale=initial_scale)

        team_performance = Normal(loc=team_skill, scale=initial_scale)

        perf_diff = tf.tile(tf.reduce_sum(team_performance, 1, keepdims=True), [1, n])
        perf_diff = perf_diff - tf.transpose(perf_diff)

        qz = Normal(loc=tf.get_variable("qz/loc", [n, 1]),
            scale=tf.nn.softplus(tf.get_variable("qz/scale", [n, 1])))

        inference = ed.KLqp({team_skill: qz}, data={perf_diff: self.data*25})
        inference.initialize(optimizer=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
            epsilon=1e-08))
        tf.global_variables_initializer().run()

        self.loss = np.empty(n_iter, dtype=np.float32)
        for i in range(n_iter):
            info_dict = inference.update()
            self.loss[i] = info_dict["loss"]
            inference.print_progress(info_dict)

        self._trained = True

        sess = ed.get_session()

        self.team_skill = sess.run(qz)
        return

    def predict(self, team1, team2):
        if self._trained:
            home = self.team_num_map[team1]
            away = self.team_num_map[team2]
            home_skill = self.team_skill[home]
            away_skill = self.team_skill[away]
            return 1 if home_skill > away_skill else -1
        else:
            raise ValueError("Model not trained")

    def predict_many(self, matchups):
        if self._trained:
            outputs = []
            for matchup in matchups:
                outputs.append(self.predict(matchup[0], matchup[1]))
            return outputs
        else:
            raise ValueError("Model not trained")

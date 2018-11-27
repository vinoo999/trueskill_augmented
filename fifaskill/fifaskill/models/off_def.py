import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import Normal, Poisson
from fifaskill.data_processing import process


class OffenseDefense(object):
    def __init__(self, data=None, goal_dif=False, n_iter=1000):
        if data is None:
            raise ValueError("Data cannot be null")

        self._trained = False
        _, _, goal_differences, team_num_map = process.win_loss_matrix(data)
        self.team_num_map = team_num_map
        self.data = goal_differences

        self.train(n_iter=n_iter)

    def train(self, n_iter=1000):
        # Model based on Alek's toy model in Jupyter
        n = len(self.team_num_map.keys())

        initial_loc = tf.ones((n, 1), dtype='float32') * 25
        initial_scale = tf.ones((n, 1),  dtype='float32') * (25/3)**2

        with tf.name_scope('model'):
            team_off = Normal(loc=initial_loc, scale=initial_scale)
            team_def = Normal(loc=initial_loc, scale=initial_scale)

            team_off_performance = Normal(loc=team_off, scale=initial_scale)
            team_def_performance = Normal(loc=team_off, scale=initial_scale)

            off_tile = tf.tile(tf.reduce_sum(
                    team_off_performance, 1, keepdims=True), [1, n])
            def_tile = tf.tile(tf.reduce_sum(
                    team_def_performance, 1, keepdims=True), [1, n])
            goals_scored_performance = off_tile - tf.transpose(def_tile)
            goals_allowed_performance = def_tile - tf.transpose(off_tile)

            goals_scored = Poisson(rate=goals_scored_performance)
            goals_allowed = Poisson(rate=goals_allowed_performance)

            perf_diff = goals_scored - goals_allowed

        with tf.name_scope('posterior'):
            qoff = Normal(loc=tf.get_variable("qoff/loc", [n, 1]),
                          scale=tf.nn.softplus(
                              tf.get_variable("qoff/scale", [n, 1])))
            qdef = Normal(loc=tf.get_variable("qdef/loc", [n, 1]),
                          scale=tf.nn.softplus(
                              tf.get_variable("qdef/scale", [n, 1])))

        inference = ed.KLqp({team_off: qoff, team_def: qdef},
                            data={perf_diff: self.data*25})
        inference.initialize(optimizer=tf.train.AdamOptimizer
                             (learning_rate=0.001, beta1=0.9, beta2=0.999,
                              epsilon=1e-08),
                             n_iter=n_iter,
                             logdir='/tmp/tensorboard_logs')
        tf.global_variables_initializer().run()

        self.loss = np.empty(n_iter, dtype=np.float32)
        for i in range(n_iter):
            info_dict = inference.update()
            self.loss[i] = info_dict["loss"]
            inference.print_progress(info_dict)

        self._trained = True

        sess = ed.get_session()
        self.team_off, self.team_def = sess.run([qoff, qdef])
        return

    def predict(self, team1, team2):
        if self._trained:
            home = self.team_num_map[team1]
            away = self.team_num_map[team2]
            home_skill = self.team_off[home] - self.team_def[away]
            away_skill = self.team_off[away] - self.team_def[home]
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

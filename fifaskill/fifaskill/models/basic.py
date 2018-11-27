import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import Normal, PointMass, Poisson, Gamma, Empirical
from fifaskill.data_processing import process


class Basic(object):
    def __init__(self, data=None, goal_dif=False, n_iter=1000):
        if data is None:
            raise ValueError("Data cannot be null")

        self._trained = False
        matchups, draws, goal_differences, team_num_map = process.win_loss_matrix(data)
        self.total_matches = np.abs(matchups) + draws
        self.team_num_map = team_num_map
        self.data = goal_differences / (self.total_matches + 1.0)

        self.train(n_iter=n_iter)

    def train(self, n_iter=1000):
        # Model based on Alek's toy model in Jupyter
        n = len(self.team_num_map.keys())

        initial_loc = tf.zeros((n, 1), dtype='float32')
        initial_scale = tf.ones((n, 1),  dtype='float32')
        # initial_scale2 =  initial_scale / 5.0

        with tf.name_scope('model'):
            team_skill = Normal(loc=initial_loc, scale=initial_scale)
            # team_skill = Gamma(concentration=initial_loc, rate=initial_scale)
            team_performance = Poisson(tf.exp(team_skill))

            # team_performance = Normal(loc=team_skill, scale=initial_scale2)

            perf_diff_tmp = tf.tile(tf.reduce_sum(team_performance, 1,
                                keepdims=True),
                                [1, n])
            perf_diff = tf.multiply((perf_diff_tmp - tf.transpose(perf_diff_tmp)), (1 / (self.total_matches+1.0)))

        with tf.name_scope('posterior'):
            # qz = Normal(loc=tf.get_variable("qz/loc", [n, 1]),
            #             scale=tf.nn.softplus(tf.get_variable("qz/scale",
            #                                                  [n, 1])))
            qz = PointMass(tf.nn.softplus(tf.Variable(tf.random_normal([n,1], mean=0, stddev=1))))
            # qz = Empirical(params=tf.Variable(tf.zeros([n_iter, n, 1])))

        inference = ed.MAP({team_skill: qz}, data={perf_diff: self.data})
        # inference.initialize(n_iter=n_iter)
        inference.initialize(optimizer=tf.train.AdamOptimizer
                             (learning_rate=0.001, beta1=0.9, beta2=0.999,
                              epsilon=1e-08),
                             n_iter=n_iter)
        tf.global_variables_initializer().run()

        # inference.run()
        self.loss = np.empty(n_iter, dtype=np.float32)
        for i in range(n_iter):
            info_dict = inference.update()
            self.loss[i] = info_dict["loss"]
            inference.print_progress(info_dict)

        self._trained = True

        sess = ed.get_session()
        self.team_skill = sess.run(qz)
        self.tmp = sess.run(perf_diff, feed_dict={team_skill: self.team_skill})

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

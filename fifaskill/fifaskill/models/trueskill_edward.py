import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import Normal, PointMass, Poisson, Gamma, Empirical
from fifaskill.data_processing import process


class Toy(object):
    def __init__(self, data=None, goal_dif=False, n_iter=1000):
        if data is None:
            raise ValueError("Data cannot be null")

        self._trained = False
        matchups, draws, goal_scored, team_num_map = process.win_loss_matrix(data, goal_scored=True)
        self.total_matches = np.abs(matchups) + draws
        self.team_num_map = team_num_map
        self.data = goal_scored

        self.train(n_iter=n_iter)

    def train(self, n_iter=1000):
        n = len(self.team_num_map.keys())

        initial_loc = tf.ones((n, 1), dtype='float32')*np.log(3)
        initial_scale = tf.ones((n, 1),  dtype='float32')*0.1
        initial_scale2 =  initial_scale / 5

        with tf.name_scope('model'):
            offense = Normal(loc=initial_loc, scale=initial_scale)
            defense = Normal(loc=initial_loc, scale=initial_scale)

            offensive_performance = Normal(loc=offense, scale=initial_scale2)
            defensive_performance = Normal(loc=defense, scale=initial_scale2)

            off_perf_cum = tf.tile(tf.reduce_sum(offensive_performance, 1,
                                keepdims=True),
                                [1, n])
            def_perf_cum = tf.tile(tf.reduce_sum(defensive_performance, 1,
                                keepdims=True),
                                [1, n])

            total_matches = tf.placeholder(tf.float32, [n,n])

            model_skill_diff = off_perf_cum - tf.transpose(def_perf_cum)

            goal_scored = Poisson(tf.exp(model_skill_diff))

            self.goal_scored_dif = tf.multiply(total_matches, goal_scored)

        with tf.name_scope('posterior'):
            # qz = Normal(loc=tf.get_variable("qz/loc", [n, 1]),
            #             scale=tf.nn.softplus(tf.get_variable("qz/scale",
            #                                                  [n, 1])))
            # qz = PointMass(tf.nn.softplus(tf.Variable(tf.random_normal([n,1], mean=2, stddev=1))))
            qo = PointMass(tf.nn.softplus(tf.Variable(tf.random_normal([n,1], mean=1, stddev=0.2))))
            qd = PointMass(tf.nn.softplus(tf.Variable(tf.random_normal([n,1], mean=1, stddev=0.2))))
            
            # qo = Normal(loc=tf.get_variable("qo/loc", [n, 1]),
            #             scale=tf.nn.softplus(tf.get_variable("qo/scale",
            #                                                  [n, 1])))
            # qd = Normal(loc=tf.get_variable("qd/loc", [n, 1]),
            #             scale=tf.nn.softplus(tf.get_variable("qd/scale",
            #                                                  [n, 1])))

            # qz = Empirical(params=tf.Variable(tf.zeros([n_iter, n, 1])))

        # inference = ed.HMC({team_skill: qz}, data={perf_diff: self.data})
        # inference = ed.KLqp({offense: qo, defense: qd}, data={self.goal_scored_cum: self.data, total_matches: self.total_matches})
        inference = ed.MAP({offense: qo, defense: qd}, data={self.goal_scored_cum: self.data, total_matches: self.total_matches})
        # inference = ed.KLqp({offense: self.qo, defense: self.qd}, data={goal_scored_cum: self.data, total_matches: self.total_matches})

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

        self.sess = ed.get_session()
        self.team_skill = self.sess.run([qo, qd])
        # self.team_skill = (offense, defense)
        # self.tmp = self.sess.run(goal_scored_cum, feed_dict={offense: self.team_skill[0], defense: self.team_skill[1], total_matches: self.total_matches})

        return

    def predict(self, team1, team2):
        if self._trained:
            home = self.team_num_map[team1]
            away = self.team_num_map[team2]
            off_skills = self.team_skill[0]
            def_skills = self.team_skill[1]
            if off_skills[home] - def_skills[away] > off_skills[away] - def_skills[home]:
                return 1
            else:
                return -1
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

import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import Normal, Poisson
from fifaskill.data_processing import process


class LinearRegressor(object):
    def __init__(self, data=None, goal_dif=False, n_iter=1000):
        if data is None:
            raise ValueError("Data cannot be null")

        self._trained = False
        matches, results, team_num_map = process.match_vectors(data)
        self.xs = matches
        self.ys = results
        self.team_num_map = team_num_map
        self.train(n_iter=n_iter)

    def train(self, n_iter=1000):
        D = len(self.team_num_map.keys())
        N = self.xs.shape[0]
        with tf.name_scope('model'):
            self.X = tf.placeholder(tf.float32, [N, D])
            self.w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
            self.b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
            self.y = Normal(loc=ed.dot(self.X, self.w) + self.b,
                            scale=tf.ones(N))

        with tf.name_scope('posterior'):
            self.qw = Normal(loc=tf.get_variable("qw/loc", [D]),
                             scale=tf.nn.softplus(tf.get_variable("qw/scale",
                                                                  [D])))
            self.qb = Normal(loc=tf.get_variable("qb/loc", [1]),
                             scale=tf.nn.softplus(tf.get_variable("qb/scale",
                                                                  [1])))

        inference = ed.ReparameterizationKLqp({self.w: self.qw,
                                               self.b: self.qb},
                                              data={self.X: self.xs,
                                                    self.y: self.ys})
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

        graph = tf.get_default_graph()
        self.team_skill = graph.get_tensor_by_name("qw/loc:0").eval()
        self.bias = graph.get_tensor_by_name("qb/loc:0").eval()

        self.y_post = ed.copy(self.y, {self.w: self.qw, self.b: self.qb})
        return

    def predict(self, team1, team2):
        if self._trained:
            home = self.team_num_map[team1]
            away = self.team_num_map[team2]
            dif = round(self.team_skill[home] - self.team_skill[away] +
                        self.bias[0])
            if dif >= 1:
                return 1
            elif dif <= -1:
                return -1
            else:
                return 0
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

    def ppc(self, T_type):
        if T_type == 'mean':
            def T(ys, xs): return tf.reduce_mean(self.y_post)
        else:
            def T(ys, xs): return tf.reduce_mean(self.y_post)
        stats = ed.ppc(T, data={self.X: self.xs, self.y_post: self.ys},
                       latent_vars={self.w: self.qw, self.b: self.qb},
                       n_samples=1000)
        return stats[0]


class TrueSkillRegressor(object):
    def __init__(self, data=None, goal_dif=False, n_iter=1000):
        if data is None:
            raise ValueError("Data cannot be null")

        self._trained = False
        matches, results, team_num_map = process.match_vectors(data)
        self.xs = matches
        self.ys = results
        self.team_num_map = team_num_map
        self.train(n_iter=n_iter)

    def train(self, n_iter=1000):
        D = len(self.team_num_map.keys())
        N = self.xs.shape[0]
        with tf.name_scope('model'):
            self.X = tf.placeholder(tf.float32, [N, D])
            self.w = Normal(loc=tf.ones(D) * 25, scale=tf.ones(D) * (25 / 3))
            self.y = Normal(loc=ed.dot(self.X, self.w), scale=tf.ones(N))

        with tf.name_scope('posterior'):
            self.qw = Normal(loc=tf.get_variable("qw/loc", [D]),
                             scale=tf.nn.softplus(tf.get_variable("qw/scale",
                                                                  [D])))

        inference = ed.ReparameterizationKLqp({self.w: self.qw},
                                              data={self.X: self.xs,
                                                    self.y: self.ys})
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

        graph = tf.get_default_graph()
        self.team_skill = graph.get_tensor_by_name("qw/loc:0").eval()
        self.perf_variance = graph.get_tensor_by_name("qw/scale:0").eval()

        self.y_post = ed.copy(self.y, {self.w: self.qw})
        return

    def predict(self, team1, team2):
        if self._trained:
            home = self.team_num_map[team1]
            away = self.team_num_map[team2]
            dif = round(self.team_skill[home] - self.team_skill[away])
            if dif >= 1:
                return 1
            elif dif <= -1:
                return -1
            else:
                return 0
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

    def ppc(self, T_type):
        if T_type == 'mean':
            def T(ys, xs): return tf.reduce_mean(self.y_post)
        else:
            def T(ys, xs): return tf.reduce_mean(self.y_post)
        stats = ed.ppc(T, data={self.X: self.xs, self.y_post: self.ys},
                       latent_vars={self.w: self.qw},
                       n_samples=1000)
        return stats[0]


class LogLinear(object):
    def __init__(self, data=None, goal_dif=False, n_iter=1000):
        if data is None:
            raise ValueError("Data cannot be null")

        self._trained = False
        matches, results, team_num_map = process.match_vectors(data)
        self.xs = matches
        self.ys = np.exp(results)
        self.team_num_map = team_num_map
        self.train(n_iter=n_iter)

    def train(self, n_iter=1000):
        D = len(self.team_num_map.keys())
        N = self.xs.shape[0]
        with tf.name_scope('model'):
            self.X = tf.placeholder(tf.float32, [N, D])
            self.w1 = Normal(loc=tf.zeros(D), scale=tf.ones(D))
            # self.b1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))
            self.y1 = Poisson(rate=tf.exp(ed.dot(self.X, self.w1)))

        with tf.name_scope('posterior'):
            self.qw1 = Normal(loc=tf.get_variable("qw1_ll/loc", [D]),
                              scale=tf.nn.softplus(tf.get_variable
                                                   ("qw1_ll/scale",
                                                    [D])))
            # self.qb1 = Normal(loc=tf.get_variable("qb1/loc", [1]),
            #                  scale=tf.nn.softplus(tf.get_variable("qb1/scale",
            #                                                        [1])))

        inference = ed.ReparameterizationKLqp({self.w1: self.qw1},
                                              data={self.X: self.xs,
                                                    self.y1: self.ys})
        inference.initialize(optimizer=tf.train.AdamOptimizer
                             (learning_rate=0.001, beta1=0.9, beta2=0.999,
                              epsilon=1e-08),
                             n_iter=n_iter)
        tf.global_variables_initializer().run()

        self.loss = np.empty(n_iter, dtype=np.float32)
        for i in range(n_iter):
            info_dict = inference.update()
            self.loss[i] = info_dict["loss"]
            inference.print_progress(info_dict)

        self._trained = True

        graph = tf.get_default_graph()
        self.team_skill = graph.get_tensor_by_name("qw1_ll/loc:0").eval()
        self.perf_variance = graph.get_tensor_by_name("qw1_ll/scale:0").eval()
        # self.bias = (graph.get_tensor_by_name("qb1/loc:0").eval(),
        #              graph.get_tensor_by_name("qb2/loc:0").eval())

        self.y_post = ed.copy(self.y1, {self.w1: self.qw1})
        return

    def predict(self, team1, team2):
        if self._trained:
            home = self.team_num_map[team1]
            away = self.team_num_map[team2]

            home_skill = self.team_skill[0][home] + self.team_skill[1][away]
            away_skill = self.team_skill[1][home] + self.team_skill[0][away]

            home_goals = np.random.poisson(lam=np.exp(home_skill))
            away_goals = np.random.poisson(lam=np.exp(away_skill))
            dif = home_goals - away_goals
            # print(home_goals, away_goals)
            if dif >= 1:
                return 1
            elif dif <= -1:
                return -1
            else:
                return 0
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

    def ppc(self, T_type):
        if T_type == 'mean':
            def T1(ys, xs): return tf.reduce_mean(self.y_post)
        else:
            def T1(ys, xs): return tf.reduce_mean(self.y_post)
        stats1 = ed.ppc(T1, data={self.X: self.xs, self.y_post: self.ys},
                        latent_vars={self.w1: self.qw1},
                        n_samples=1000)
        return stats1


class LogLinearOffDef(object):
    def __init__(self, data=None, goal_dif=False, n_iter=1000):
        if data is None:
            raise ValueError("Data cannot be null")

        self._trained = False
        matches, results, team_num_map = process.match_vectors(data,
                                                               goals=True)
        self.xs = matches
        self.ys = results
        self.team_num_map = team_num_map
        self.train(n_iter=n_iter)

    def train(self, n_iter=1000):
        D = len(self.team_num_map.keys())
        N = self.xs.shape[0]
        with tf.name_scope('model'):
            self.X = tf.placeholder(tf.float32, [N, D])
            self.w1 = Normal(loc=tf.zeros(D), scale=tf.ones(D))
            self.b1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))
            self.y1 = Poisson(rate=tf.exp(ed.dot(self.X, self.w1) + self.b1))
            self.w2 = Normal(loc=tf.zeros(D), scale=tf.ones(D))
            self.b2 = Normal(loc=tf.zeros(1), scale=tf.ones(1))
            self.y2 = Poisson(rate=tf.exp(ed.dot(self.X, self.w2) + self.b2))

        with tf.name_scope('posterior'):
            self.qw1 = Normal(loc=tf.get_variable("qw1_llod/loc", [D]),
                              scale=tf.nn.softplus(tf.get_variable
                                                   ("qw1_llod/scale",
                                                    [D])))
            self.qb1 = Normal(loc=tf.get_variable("qb1_llod/loc", [1]),
                              scale=tf.nn.softplus(tf.get_variable
                                                   ("qb1_llod/scale",
                                                    [1])))
            self.qw2 = Normal(loc=tf.get_variable("qw2_llod/loc", [D]),
                              scale=tf.nn.softplus(tf.get_variable
                                                   ("qw2_llod/scale",
                                                    [D])))
            self.qb2 = Normal(loc=tf.get_variable("qb2_llod/loc", [1]),
                              scale=tf.nn.softplus(tf.get_variable
                                                   ("qb2_llod/scale",
                                                    [1])))

        inference = ed.ReparameterizationKLqp({self.w1: self.qw1,
                                               self.b1: self.qb1,
                                               self.w2: self.qw2,
                                               self.b2: self.qb2},
                                              data={self.X: self.xs,
                                                    self.y1: self.ys[:, 0],
                                                    self.y2: self.ys[:, 1]})
        inference.initialize(optimizer=tf.train.AdamOptimizer
                             (learning_rate=0.001, beta1=0.9, beta2=0.999,
                              epsilon=1e-08),
                             n_iter=n_iter)
        tf.global_variables_initializer().run()

        self.loss = np.empty(n_iter, dtype=np.float32)
        for i in range(n_iter):
            info_dict = inference.update()
            self.loss[i] = info_dict["loss"]
            inference.print_progress(info_dict)

        self._trained = True

        graph = tf.get_default_graph()
        self.team_skill = (graph.get_tensor_by_name("qw1_llod/loc:0").eval(),
                           graph.get_tensor_by_name("qw2_llod/loc:0").eval())
        self.bias = (graph.get_tensor_by_name("qb1_llod/loc:0").eval(),
                     graph.get_tensor_by_name("qb2_llod/loc:0").eval())

        self.y_post = (ed.copy(self.y1, {self.w1: self.qw1,
                                         self.b1: self.qb1}),
                       ed.copy(self.y2, {self.w2: self.qw2,
                                         self.b2: self.qb2}))
        return

    def predict(self, team1, team2):
        if self._trained:
            home = self.team_num_map[team1]
            away = self.team_num_map[team2]

            home_skill = self.team_skill[0][home] + self.team_skill[1][away]
            away_skill = self.team_skill[1][home] + self.team_skill[0][away]

            home_goals = np.random.poisson(lam=np.exp(home_skill))
            away_goals = np.random.poisson(lam=np.exp(away_skill))
            dif = home_goals - away_goals
            print(home_goals, away_goals)
            if dif >= 1:
                return 1
            elif dif <= -1:
                return -1
            else:
                return 0
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

    def ppc(self, T_type):
        if T_type == 'mean':
            def T1(ys, xs): return tf.reduce_mean(self.y_post[0])

            def T2(ys, xs): return tf.reduce_mean(self.y_post[1])

        else:
            def T1(ys, xs): return tf.reduce_mean(self.y_post[0])

            def T2(ys, xs): return tf.reduce_mean(self.y_post[1])

        stats1 = ed.ppc(T1, data={self.X: self.xs,
                                  self.y_post[0]: self.ys[:, 0]},
                        latent_vars={self.w1: self.qw1, self.b1: self.qb1},
                        n_samples=1000)
        stats2 = ed.ppc(T2, data={self.X: self.xs,
                                  self.y_post[1]: self.ys[:, 1]},
                        latent_vars={self.w2: self.qw2, self.b2: self.qb2},
                        n_samples=1000)
        return (stats1[0], stats2[0])

import gym_adversarialgrid.agents.exp3mg as exp3mg
from collections import defaultdict
import math


class HedgeMG(exp3mg.Exp3MG):
    """
    Extends a version of Hedge (Freund&Schapire1995) with the notion of state.
    The version extended here is based on Auer et. al 2002

    References:

    Freund, Y., & Schapire, R. E. (1995). A desicion-theoretic generalization of
    on-line learning and an application to boosting, 55(1), 23–37.
    https://doi.org/10.1007/3-540-59119-2_166

    Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002).
    The nonstochastic multiarmed bandit problem.
    Society for Industrial and Applied Mathematics, 32(1), 48–77.
    """

    def __init__(self, *args, **kwargs):
        super(HedgeMG, self).__init__(*args, **kwargs)

        self.config['gamma'] = 0.07  # inits with a default value

        self.config.update(kwargs)

        self.weights = defaultdict(
            lambda: [1.0 / self.action_space.n] * self.action_space.n
        )

        # policy initialized as uniformly random
        self.policy = defaultdict(lambda: [1.0 / self.action_space.n] * self.action_space.n)

    def learn(self, s, a, reward, sprime, done):
        # aliases:
        pi_sp = self.calculate_policy(sprime)
        q = self.q  # alias for the action value function
        n = self.action_space.n  # the number of actions
        lrn_rate = self.config['learning_rate']
        discount = self.config['discount']
        gamma = self.config['gamma']

        # estimation of the expected value of s' -- it is zero if current state is terminal
        future = sum([pi_sp[aprime] * value for aprime, value in enumerate(q[sprime])]) if not done else 0

        # minimax-Q-like update:
        q[s][a] = q[s][a] + lrn_rate * (reward + discount * future - q[s][a])

        for action in range(self.action_space.n):
            x = q[s][action]

            # scales x to [0, 1] - assuming minimum reward is -1 and max reward is +1
            # rescaling as per https://en.wikipedia.org/wiki/Feature_scaling#Rescaling
            max_x = 1  # + self.discount
            min_x = -1  # - self.discount
            scaled_x = (x - min_x) / (max_x - min_x)

            if not 0 <= scaled_x <= 1:
                print("WARNING! scaled_x=%f out of bounds!" % scaled_x)

            # finally updates the weight
            # print('q(s,a), r, f, x, ~x = %.3f, %3f, %.3f, %.3f, %.3f' % (self.q[s][a], future, reward, x, scaled_x))
            self.weights[s][action] *= math.exp(gamma * scaled_x / n)
            # print(self.weights[s])


class HedgeMG_1995(HedgeMG):
    """
    Extends the original Hedge (Freund&Schapiro 1995)  with the notion of state.

    Reference:
    Freund, Y., & Schapire, R. E. (1995). A desicion-theoretic generalization of
    on-line learning and an application to boosting, 55(1), 23–37.
    https://doi.org/10.1007/3-540-59119-2_166
    """

    def __init__(self, *args, **kwargs):
        super(HedgeMG_1995, self).__init__(*args, **kwargs)

        self.config['alpha'] = 0.2  # sets a default value
        self.config.update(kwargs)
        print("Params: %s", self.config)

        self.weights = defaultdict(
            lambda: [1.0 / self.action_space.n] * self.action_space.n
        )

    def learn(self, s, a, reward, sprime, done):
        # aliases:
        pi_sp = self.calculate_policy(sprime)
        q = self.q  # alias for the action value function
        n = self.action_space.n  # the number of actions
        discount = self.config['discount']
        lrn_rate = self.config['learning_rate']

        # estimation of the expected value of s' -- it is zero if current state is terminal
        future = sum([pi_sp[aprime] * value for aprime, value in enumerate(q[sprime])]) if not done else 0

        # minimax-Q-like update:
        q[s][a] = q[s][a] + lrn_rate * (reward + discount * future - q[s][a])

        for action in range(self.action_space.n):
            x = q[s][action]

            # scales x to [0, 1] - assuming minimum reward is -1 and max reward is +1
            # rescaling as per https://en.wikipedia.org/wiki/Feature_scaling#Rescaling
            max_x = 1  # + self.discount
            min_x = -1  # - self.discount
            scaled_x = (x - min_x) / (max_x - min_x)

            if not 0 <= scaled_x <= 1:
                print("WARNING! scaled_x=%f out of bounds!" % scaled_x)

            # finally updates the weight
            # print('q(s,a), r, f, x, ~x = %.3f, %3f, %.3f, %.3f, %.3f' % (self.q[s][a], future, reward, x, scaled_x))
            self.weights[s][action] += scaled_x
            # print(self.weights[s])

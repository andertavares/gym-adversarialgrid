import gym_adversarialgrid.agents.tabular as tabular
from gym_adversarialgrid.agents.util import categorical_draw
from collections import defaultdict
import math
import numpy as np


class Exp3MG(tabular.TabularQAgent):
    """
    Extends Exp3 (Auer et. al 1995) with the notion of state.
    The implementation of Exp3 we are extending is the one shown in Auer et. al 2002.

    References:

    Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (1995).
    Gambling in a rigged casino: The adversarial multi-armed bandit problem.
    Proceedings of IEEE 36th Annual Foundations of Computer Science, 322–331.
    https://doi.org/10.1109/SFCS.1995.492488

    Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002).
    The nonstochastic multiarmed bandit problem.
    Society for Industrial and Applied Mathematics, 32(1), 48–77.

    TODO: replace categorical draw with multinomial: np.random.multinomial(num_draws, probabilities)
    or np.random.choice:

    aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
    np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])

    TODO: read parameters from config dict
    """

    def __init__(self, *args, **kwargs):
        super(Exp3MG, self).__init__(*args, **kwargs)

        self.config['gamma'] = 0.07  # inits with a default value kwargs['gamma'] if 'gamma' in kwargs else 0.07
        # self.lrn_rate = kwargs['lrn_rate'] if 'lrn_rate' in kwargs else 0.1
        # self.discount = kwargs['discount'] if 'discount' in kwargs else 0.9

        self.config.update(kwargs)

        n_actions = self.action_space.n

        # lazy initialization -- each state will be assigned a vector of ones in the first time
        self.q = defaultdict(
            # lambda: [0.01] * n_actions
            lambda: [1.] * n_actions
        )

        self.weights = defaultdict(
            lambda: [1.] * n_actions
        )

        # policy initialized as uniformly random
        self.policy = defaultdict(lambda: [1.0 / n_actions] * n_actions)
        # self.policy = [self.calculate_policy(s) for s in self.observation]

    def calculate_policy(self, state):
        """
        Calculates the policy for a given state and returns it
        :param state:
        :return: list(float) the policy (probability vector) for that state
        """
        # short aliases
        s = state  # s stands for state
        g = self.config['gamma']  # g stands for gamma
        n = self.action_space.n  # n stands for the number of actions
        pi_s = self.policy[state]  # pi_s stands for the policy in state s

        sum_weights = sum(self.weights[s])

        # the policy is a probability vector, giving the probability of each action
        pi_s = [((1 - g) * w / sum_weights) + (g / n) for w in self.weights[s]]
        # print(state, pi_s)
        return pi_s

    def act(self, observation):
        prob_vector = self.calculate_policy(observation)
        return categorical_draw(prob_vector)

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

        # q is then fed as Exp3's reward -- x is a value to be scaled and weighted by its probability
        x = q[s][a]

        # scales x to [0, 1] - assuming minimum reward is -1 and max reward is +1
        # rescaling as per https://en.wikipedia.org/wiki/Feature_scaling#Rescaling
        max_x = 1  # + self.discount
        min_x = -1  # - self.discount

        scaled_x = (x - min_x) / (max_x - min_x)

        if not 0 <= scaled_x <= 1:
            print("WARNING! scaled_x=%f out of bounds!" % scaled_x)

        # weights the value by its probability -- estimates the reward
        x_hat = scaled_x / self.policy[s][a]

        # finally updates the weight
        # print('q(s,a), r, f, x, ~x, ^x = %.3f, %3f, %.3f, %.3f, %.3f, %.3f' % (self.q[s][a], future, reward, x, scaled_x, x_hat))
        self.weights[s][a] *= math.exp(gamma * x_hat / n)

    def greedy_policy(self):
        """
        Returns the greedy (deterministic) policy of this agent.
        Differently from regular Tabular agents, the greedy policy is related to the weights, not the
        action-value function
        :return:
        """
        # print(self.weights)
        policy = defaultdict(lambda: 0)

        for entry, values in self.weights.items():
            policy[entry] = np.argmax(self.weights[entry])
            # print(policy)

        return policy


class Exp3MG_1995(Exp3MG):
    """
    Extends the original Exp3 (Auer et. al 1995) with the notion of state.

    References:

    Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (1995).
    Gambling in a rigged casino: The adversarial multi-armed bandit problem.
    Proceedings of IEEE 36th Annual Foundations of Computer Science, 322–331.
    https://doi.org/10.1109/SFCS.1995.492488
    """

    def __init__(self, *args, **kwargs):
        super(Exp3MG_1995, self).__init__(*args, **kwargs)

        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.2

    def calculate_policy(self, state):
        """
        Calculates the policy for a given state and returns it
        :param state:
        :return: list(float) the policy (probability vector) for that state
        """
        # short aliases
        s = state  # s stands for state
        g = self.gamma  # g stands for gamma
        n = self.action_space.n  # n stands for the number of actions
        a = self.alpha
        pi_s = self.policy[state]  # pi_s stands for the policy in state s
        weights = self.weights[state]
        # print(weights)


        # obtains the probability vector from Hedge: p_i(t) = (1+alpha)^s_i(t) / sum_{j \in K} (1+alpha)^s_j(t)
        sum_weights_exponentials = sum([(1 + a) ** w for w in weights])
        pre_prob = [(((1 + a) ** w) / sum_weights_exponentials) for w in weights]

        # the policy is a probability vector, giving the probability of each action
        pi_s = [((1 - g) * p) + (g / n) for p in pre_prob]

        return pi_s

    def learn(self, s, a, reward, sprime, done):
        # aliases:
        pi_sp = self.calculate_policy(sprime)
        q = self.q  # alias for the action value function
        n = self.action_space.n  # the number of actions

        # estimation of the expected value of s' -- it is zero if current state is terminal
        future = sum([pi_sp[aprime] * value for aprime, value in enumerate(q[sprime])]) if not done else 0

        # minimax-Q-like update:
        q[s][a] = q[s][a] + self.lrn_rate * (reward + self.discount * future - q[s][a])

        x = q[s][a]

        # scales x to [0, 1] - assuming minimum reward is -1 and max reward is +1
        # rescaling as per https://en.wikipedia.org/wiki/Feature_scaling#Rescaling
        max_x = 1  # + self.discount
        min_x = -1  # - self.discount
        scaled_x = (x - min_x) / (max_x - min_x)

        if not 0 <= scaled_x <= 1:
            print("WARNING! scaled_x=%f out of bounds!" % scaled_x)

        # weights the value by its probability -- estimates the reward...
        x_hat = (self.gamma / n) * (scaled_x / self.policy[s][a])

        # finally updates the weight
        # print('q(s,a), r, f, x, ~x, ^x = %.3f, %3f, %.3f, %.3f, %.3f, %.3f' % (self.q[s][a], future, reward, x, scaled_x, x_hat))
        self.weights[s][a] += x_hat

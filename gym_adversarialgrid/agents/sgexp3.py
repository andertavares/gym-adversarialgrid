import gym_adversarialgrid.agents.tabular as tabular
from collections import defaultdict
import random
import math


def categorical_draw(probabilities):
    """
    Selects an option with a roulette-like process
    :param probabilities:
    :return:
    """
    z = random.random()
    cum_prob = 0.0

    for choice, prob in enumerate(probabilities):
        cum_prob += prob
        if cum_prob > z:
            return choice

    print('Warning: categorical_draw reached its end')
    return len(probabilities) - 1  # I think code should not reach here


class SGExp3(tabular.TabularQAgent):
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
    """

    def __init__(self, *args, **kwargs):
        super(SGExp3, self).__init__(*args, **kwargs)

        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.2
        self.discount = kwargs['discount'] if 'discount' in kwargs else 0.9

        n_actions = self.action_space.n

        # cannot initialize q with zeroes
        self.q = defaultdict(
            lambda: [0.01] * n_actions
        )

        # policy initialized as uniformly random
        self.policy = defaultdict(lambda: [1.0 / n_actions] * n_actions)

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
        pi_s = self.policy[state]  # pi_s stands for the policy in state s

        sum_weights = sum(self.q[s])

        # the policy is a probability vector, giving the probability of each action
        # pi(s, . ) = [(1 - gamma)*q(s,a) + gamma / n] - for each action
        # print(state, pi_s, self.q[s])
        pi_s = [((1 - g) * value / sum_weights) + (g / n) for value in self.q[s]]
        # print(state, pi_s)
        return pi_s

    def act(self, observation):
        prob_vector = self.calculate_policy(observation)
        return categorical_draw(prob_vector)

    def learn(self, s, a, reward, sprime, done):
        # aliases:
        pi_sp = self.policy[sprime]  # the policy for the next state
        q_sp = self.q[sprime]  # the action values for next state
        n = self.action_space.n  # the number of actions

        # value of next state, it is zero if current state is terminal
        future = sum([pi_sp[ap] * value for ap, value in enumerate(q_sp)]) if not done else 0

        # x is a value to be scaled and weighted by its probability
        x = reward + self.discount * future

        # scales x to [0, 1] - assuming minimum reward is -1 and max reward is +1
        # rescaling as per https://en.wikipedia.org/wiki/Feature_scaling#Rescaling
        max_x = 1 + self.discount
        min_x = -1 - self.discount

        scaled_x = (x - min_x) / (max_x - min_x)

        # weights the value by its probability
        x_hat = scaled_x / self.policy[s][a]

        # finally updates the value
        self.q[s][a] = self.q[s][a] * math.exp(self.gamma * x_hat / n)

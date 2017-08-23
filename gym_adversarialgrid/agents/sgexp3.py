import gym_adversarialgrid.agents.tabular as tabular
import random


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
    return choice  # I think code should not reach here


class SGExp3(tabular.TabularQAgent):
    def __init__(self, *args, **kwargs):
        super(SGExp3, self).__init__(*args, **kwargs)

        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.2
        self.alpha = kwargs['exp3_alpha'] if 'exp3_alpha' in kwargs else 0.5

    def act(self, observation):
        # performs selection
        # total_weight = sum(self.weights.values())
        total_prob_factor = sum(self.prob_factor(w) for w in self.weights.values())

        # the number of options in this state
        n_arms = len(self.action_space)

        # initializes the list of probabilities
        probs = []

        weights = self.q[observation]

        for arm in range(len(self.action_space)):
            # probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            # probs[arm] += self.gamma * (1.0 / float(n_arms))
            pre_prob = self.prob_factor(weights[arm]) / total_prob_factor

            # mixes pre_prob with an uniform distribution
            # gamma is the 'fraction' of uniform that gets mixed in
            probs.append((1.0 - self.gamma) * pre_prob + self.gamma / float(n_arms))

        return categorical_draw(probs)

    def learn(self, s, a, reward, sprime, done):
        # SO FAR IT IS THE PLAIN EXP3 UPDATE RULE
        n_arms = len(self.action_space)
        weights = self.q[s]

        # total_weight = sum(self.weights.values())
        total_prob_factor = sum(self.prob_factor(w) for w in weights.values())

        # rescales reward to [0,1], the original is either -1 or 1
        scaled_rwd = (reward + 1) / 2

        pre_prob = self.prob_factor(weights[a]) / total_prob_factor

        prob_of_chosen_arm = (1 - self.gamma) * pre_prob + self.gamma / n_arms
        # prob_of_chosen_arm = (1 - self.gamma) * (self.weights[chosen_arm] / total_weight) \
        #                     + self.gamma * (1.0 / float(n_arms))

        # x = reward / prob_of_chosen_arm  # probs[chosen_arm]
        mixed_rwd = (self.gamma / n_arms) * (reward / prob_of_chosen_arm)  # probs[chosen_arm]

        # growth_factor = math.exp((self.gamma / n_arms) * x)
        # self.weights[chosen_arm] *= growth_factor
        weights[a] += mixed_rwd

    def prob_factor(self, weight):
        """
        Uses the probability factor formula of Exp3
        :param weight:
        :return:
        """
        return (1 + self.alpha) ** weight

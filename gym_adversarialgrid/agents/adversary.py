from gym_adversarialgrid.agents.agent import LearningAgent
from collections import defaultdict

class Random(LearningAgent):
    """
    Has a random uniform policy

    """

    def __init__(self, *args, **kwargs):
        super(Random, self).__init__(*args, **kwargs)

    def act(self, observation):
        """
        Returns an action at random
        :param observation:
        :return:
        """
        return self.action_space.sample()

    def learn(self, s, a, reward, sprime, done):
        """
        This method does nothing. This agent does not learn.
        :param s:
        :param a:
        :param reward:
        :param sprime:
        :param done:
        :return:
        """
        pass

    def greedy_policy(self):
        """
        There is no 'greedy policy' for a random method,
        all actions have the same probability
        :return:
        """
        return defaultdict(lambda: 0)


class Fixed(LearningAgent):
    """
    An agent that always return the same action
    """

    def __init__(self, *args, **kwargs):
        """
        Always select a fixed action
        :param action:
        """
        super(Fixed, self).__init__(*args, **kwargs)
        self.action = kwargs['action']

    def act(self, observation):
        return self.action

    def learn(self, s, a, reward, sprime, done):
        """
        This method does nothing. This agent does not learn.
        :param s:
        :param a:
        :param reward:
        :param sprime:
        :param done:
        :return:
        """
        pass

    def greedy_policy(self):
        return defaultdict(lambda: self.action)

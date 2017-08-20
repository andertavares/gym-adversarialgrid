class Agent(object):
    def act(self, observation):
        """
        Acts upon an observation
        :param observation:
        :return:
        """
        raise NotImplemented

    def learn(self, s, a, reward, sprime, done):
        """
        Learns upon an experience tuple
        :param s: observation
        :param a: action
        :param reward: reward of taking a in s
        :param sprime: observation after taking a in s
        :param done: is sprime a terminal state?
        :return:
        """
        pass

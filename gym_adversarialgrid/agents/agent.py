class Agent(object):
    def __init__(self, observation_space, action_space, **userconfig):
        """
        Initializes the agent
        :param observation_space: an instance of gym.spaces.Discrete
        :param action_space: an instance of gym.spaces.Discrete
        :param userconfig: additional parameters (param=value)
        """
        self.observation_space = observation_space
        self.action_space = action_space

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

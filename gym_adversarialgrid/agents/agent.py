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


class LearningAgent(Agent):
    """
    Implements a learning agent

    """

    def __init__(self, *args, **kwargs):
        super(LearningAgent, self).__init__(*args, **kwargs)

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
        raise NotImplemented

    def train(self, env, steps):
        """
        Trains the agent for the given number of steps.
        At each step, the agent receives the current state, acts,
        receives the reward, the next state and learns upon this
        experience tuple
        :param env:
        :param steps:
        :return:
        """
        observation = env.reset()
        for t in range(steps):
            action = self.act(observation)
            next_obs, reward, done, _ = env.step(action)
            self.learn(observation, action, reward, next_obs, done)
            observation = next_obs if not done else env.reset()

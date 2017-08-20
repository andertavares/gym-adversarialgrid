from agent import Agent


class Benign(Agent):
    """
    A benign adversary that does not act
    """

    def act(self, observation):
        return 0  # noop


class Random(Agent):
    """
    Has a random uniform policy
    """

    def __init__(self, action_space):
        """
        Initializes a random adversary
        :param action_space: an instance of gym.spaces.Discrete
        """
        self.action_space = action_space

    def act(self, observation):
        """
        Try to drive the agent to the hole by deflecting
        or inverting some of its obvious actions
        :param observation:
        :return:
        """
        return self.action_space.sample()


class Fixed(Agent):
    """
    An agent that always return the same action
    """

    def __init__(self, action=0):
        """
        Always select a fixed action
        :param action:
        """
        self.action = action

    def act(self, observation):
        return self.action

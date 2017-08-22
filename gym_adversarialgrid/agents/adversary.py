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

    def act(self, observation):
        """
        Returns an action at random
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

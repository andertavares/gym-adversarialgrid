from gym_adversarialgrid.agents.agent import Agent


class Random(Agent):
    """
    Has a random uniform policy

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        """
        Always select a fixed action
        :param action:
        """
        super().__init__(*args, **kwargs)
        self.action = kwargs['action']

    def act(self, observation):
        return self.action

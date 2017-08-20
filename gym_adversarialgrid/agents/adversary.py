class Benign(object):
    """
    A benign adversary that does not act
    """

    def act(self, observation):
        return 0  # noop


class Stationary(object):
    """
    Has a fixed, stationary (stochastic) policy
    """

    def __init__(self, action_space, world):
        self.world = world
        self.nrows = len(world)
        self.ncols = len(world[0])

    def act(self, observation):
        """
        Try to drive the agent to the hole by deflecting
        or inverting some of its obvious actions
        :param observation:
        :return:
        """

# A tabular Q-learning agent
import gym.spaces.discrete as discrete
from collections import defaultdict
import numpy as np
import gym_adversarialgrid.agents.agent as agent


class TabularQAgent(agent.LearningAgent):
    """
    Agent implementing tabular Q-learning.

    """

    def __init__(self, observation_space, action_space, **userconfig):
        super(TabularQAgent, self).__init__(observation_space, action_space, **userconfig)
        if not isinstance(observation_space, discrete.Discrete):
            raise UnsupportedSpace(
                'Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(
                    observation_space, self))
        if not isinstance(action_space, discrete.Discrete):
            raise UnsupportedSpace(
                'Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space,
                                                                                                       self))
        # self.observation_space = observation_space
        # self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean": 0.0,  # Initialize Q values with this mean
            "init_std": 0.0,  # Initialize Q values with this standard deviation
            "learning_rate": 0.1,
            "eps": 0.05,  # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 10000}  # Number of iterations
        self.config.update(userconfig)
        self.q = defaultdict(
            lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"]
        )

    def act(self, observation):
        """
        Selects an action via epsilon greedy
        If action values are equal, ties are broken randomly
        :param observation: current 'state'

        """

        # epsilon greedy
        action = self.action_space.sample()
        if np.random.random() > self.config["eps"]:
            # an argmax that breaks ties randomly: https://stackoverflow.com/a/42071648
            b = self.q[observation]
            action = np.random.choice(np.flatnonzero(b == b.max()))

        return action

    def greedy_policy(self):
        """
        Returns the greedy (deterministic) policy of this agent.
        Differently from self.act, this method breaks ties deterministically via np.argmax
        :return:
        """

        policy = defaultdict(lambda: 0)

        for entry, values in self.q.items():
            policy[entry] = np.argmax(self.q[entry])

        return policy

    def learn(self, s, a, reward, sprime, done):
        q = self.q  # alias

        # value of 'future' state
        future = np.max(self.q[sprime]) if not done else 0.0

        # applies update rule: Q(s,a) = Q(s,a) + alpha(r + gamma* max_{a'}Q(s',a') - Q(s,a))
        newq = q[s][a] + self.config["learning_rate"] * (reward + self.config["discount"] * future - q[s][a])
        q[s][a] = newq

import gym_adversarialgrid.agents.tabular as tabular
from collections import defaultdict
import numpy as np
import nash
import scipy


class MinimaxQ(tabular.TabularQAgent):
    """
    Implementation of MinimaxQ (Littman1994)

    Reference: Littman, M. L. (1994). Markov games as a framework for
    multi-agent reinforcement learning. Proceedings of the International
    Conference on Machine Learning, 157(1), 157–163.
    https://doi.org/10.1.1.48.8623

    TODO: check why it goes so bad against the deterministic opponent
    """

    def __init__(self, opp_action_space, *args, **kwargs):
        super(MinimaxQ, self).__init__(*args, **kwargs)

        max_rwd = kwargs['maxrwd'] if 'maxrwd' in kwargs else 1.0

        self.opp_action_space = opp_action_space

        # initializes action values optimistically
        # outer level must be a comprehension, otherwise matrix rows will only be a reference to the first
        self.q = defaultdict(
            lambda: [[max_rwd] * opp_action_space.n for _ in range(self.action_space.n)]
        )

        # initializes the state values also optimistically
        self.v = defaultdict(lambda: max_rwd)

        # initializes the policy (equiprobable)
        self.policy = defaultdict(
            lambda: [1.0 / self.action_space.n] * self.action_space.n
        )

    def act(self, observation):
        the_policy = self.policy[observation]

        # epsilon...
        action = self.action_space.sample()

        # ...greedy
        if np.random.random() > self.config['eps']:
            try:
                action = np.random.choice(self.action_space.n, 1, p=the_policy)[0]
            except ValueError:
                print("ERROR: probabilities don't add to 1 in: %s" % the_policy)
                print("Choosing randomly")

        return action

    def learn(self, s, a, o, reward, sprime, done):
        # aliases
        q = self.q
        alpha = self.config['learning_rate']
        gamma = self.config['discount']

        # determines the value of future state and updates the action-value function q
        q[s][a][o] += alpha * (reward + gamma * self.v[sprime] - q[s][a][o])

        # updates the policy and state-value function for this state
        self.update_policy(s)

    def update_policy(self, state):
        """
        Updates the policy and value for a given state.
        Solves Q to Nash equilibrium based on Daniel Kneipp's code on StarcraftNash
        TODO: take code with starcraftnash structures out
        :param state:
        :return:
        """

        solve_vertex_prepared = lambda game, e: solve_vertex(
            game,
            mark_qmatrix_to_dump,
            mark_qmatrix_to_dump
        )
        solve_lemke_prepared = lambda game, e: solve_lemke(
            game,
            solve_vertex_prepared,
            solve_vertex_prepared,
            solve_vertex_prepared,
            solve_vertex_prepared
        )

        # solves the game using Gambit's functions
        matrix = np.array(self.q[state])  # np.array([v.values() for k, v in self.q[state].items()])
        game = nash.Game(matrix)
        # policy = solve_support(game, solve_lemke_prepared)
        policy = None
        # tries first with support enumeration:
        try:
            policy = np.absolute(list(game.support_enumeration())[0][0].round(9))
        except RuntimeError as e:
            print("Could not solve with support enumeration, will try another method.", e)

            try:
                policy = np.absolute(list(game.vertex_enumeration())[0][0].round(9))
            # except (scipy.spatial.qhull.QhullError, IndexError) as e:
            except RuntimeError as e:
                print("Could not solve with vertex enumeration, will try another method.", e)

                try:
                    policy = np.absolute(list(game.lemke_howson(0))[0].round(9))
                except Exception as e:
                    print("Could not solve with Lemke-Howson", e)
                if np.NaN in policy:
                    print("There is a NaN on the equilibria")
                    policy = None
                if all(i == 0 for i in policy):
                    print('All equilibria values are 0')
                    policy = None
                if self.action_space.n != len(policy):
                    print("Policy length does not match number of actions!")
                    policy = None

        if policy is not None:
            # updates the policy for the given state
            self.policy[state] = policy

            # creates aliases:
            n_actions = self.action_space.n
            pi = self.policy
            s = state
            n_opp_actions = self.opp_action_space.n

            # updates the value function for the given state
            # it is determined by the opponent action that minimizes my expected action value
            self.v[s] = min([sum([self.q[s][a][o] * pi[s][a] for a in range(n_actions)]) for o in range(n_opp_actions)])

        else:
            print("WARNING: Malformed policy calculated. Won't update")

    def greedy_policy(self):
        """
        Returns the greedy (deterministic) policy of this agent via argmax in each state
        :return:
        """
        # print(self.weights)
        policy = defaultdict(lambda: 0)

        for state, values in self.policy.items():
            policy[state] = np.argmax(values)

        return policy

    def train(self, env, steps):
        """
        Overrides the usual train function, to retrieve the opponent's action
        At each step, the agent receives the current state, acts,
        receives the reward, the next state, extracts opponent action and
        learns upon this experience tuple
        :param env:
        :param steps:
        :return:
        """
        observation = env.reset()
        for t in range(steps):
            action = self.act(observation)
            next_obs, reward, done, info = env.step(action)
            opp_action = info['o_idx']
            self.learn(observation, action, opp_action, reward, next_obs, done)
            observation = next_obs if not done else env.reset()

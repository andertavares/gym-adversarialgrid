import gym
import gym.spaces
import sys
import numpy as np
from gym import utils
from six import StringIO
from collections import defaultdict
import gym_adversarialgrid.envs.grid as grid
import gym_adversarialgrid.agents.adversary as adversary
import gym_adversarialgrid.agents.tabular as tabular
import gym_adversarialgrid.agents.minimaxq as minimaxq
import gym_adversarialgrid.agents.hedgemg as hedgemg

# actions for the adversary
NOOP = 0
INVERT = 1
DEFLECT = 2

OPPONENTS = {
    "Random": adversary.Random,
    "Fixed": adversary.Fixed,
    "QLearning": tabular.TabularQAgent,
    "MinimaxQ": minimaxq.MinimaxQ,
    "Hedge": hedgemg.HedgeMG,
}


class AdversarialGrid(grid.Grid):
    """
    Inspired in frozen lake. The world is like:

        S___
        _H_H
        ___H
        H__G

    S : starting point, safe
    _ : frozen surface, safe
    H : hole, fall to your doom
    G : goal, yay!

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, -1 if fall in a hole and -0.01 otherwise.
    There's an adversary that disturbs some moves you make.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    opponent_action_names = {
        NOOP: "No-op",
        DEFLECT: "Deflect",
        INVERT: "Invert"
    }

    """
    This array can be used to print the greedy policy of the opponent
    """
    opponent_action_desc = {
        NOOP: "-",
        DEFLECT: "D",
        INVERT: "I"
    }

    # no-ops don't change action
    no_ops = {
        grid.LEFT: grid.LEFT,
        grid.DOWN: grid.DOWN,
        grid.RIGHT: grid.RIGHT,
        grid.UP: grid.UP,
        grid.STAY: grid.STAY,
    }

    # deflections add 90 degrees to intended direction
    deflections = {
        grid.LEFT: grid.DOWN,
        grid.DOWN: grid.RIGHT,
        grid.RIGHT: grid.UP,
        grid.UP: grid.LEFT,
        grid.STAY: grid.STAY  # cannot deflect a no-op
    }

    # inverts intended direction
    inversions = {
        grid.LEFT: grid.RIGHT,
        grid.DOWN: grid.UP,
        grid.RIGHT: grid.LEFT,
        grid.UP: grid.DOWN,
        grid.STAY: grid.STAY,  # cannot invert a stay
    }

    # opponent action indexes this dict
    opponent_action_effects = {
        NOOP: no_ops,
        DEFLECT: deflections,
        INVERT: inversions
    }

    def __init__(self, opponent='Random', *args, **kwargs):
        params = {
            'desc': None,
            'map': '3x4',
        }

        params.update(kwargs)

        super(AdversarialGrid, self).__init__(params['desc'], params['map'])

        self.opp_name = opponent

        # 3 actions: NOOP/DEFLECT/INVERT
        self.opp_action_space = gym.spaces.Discrete(
            len(self.opponent_action_names.values())
        )

        # MinimaxQ has a special treatment:
        if opponent == 'MinimaxQ':
            # opp_action space and action_space are swapped because MMQ must play the opponent
            self.opponent = minimaxq.MinimaxQ(
                self.observation_space, self.opp_action_space,
                self.action_space, **kwargs
            )

        else:
            self.opponent = OPPONENTS[opponent](self.observation_space, self.opp_action_space, **kwargs)

    def print_opponent_greedy_policy(self, outstream=sys.stdout):
        """
        Prints the opponent greedy policy
        :return:
        """

        self.print_deterministic_policy(
            self.opponent.greedy_policy(),
            action_names=self.opponent_action_desc
        )

    def print_combined_policies(self, agent):
        agent_policy = agent.greedy_policy()

        resulting_policy = defaultdict(lambda: 0)

        # determines the agent and opponent actions for each state
        # a stands for agent action and o for opponent's
        for state, policy in agent_policy.items():
            a = np.argmax(agent.q[state])
            o = np.argmax(self.opponent.q[state])
            resulting_policy[state] = self.opponent_action_effects[o][a]

        self.print_deterministic_policy(
            resulting_policy
        )

    def _step(self, a):
        """
        Receives the action of the agent and returns the outcome
        :param a:the action to execute (an integer)
        :return: tuple(state, reward, done, info)
        """
        # stores current state
        s = self.current_state

        # opponent's action
        o = self.opponent.act(self.current_state)

        resulting_action = self.opponent_action_effects[o][a]

        # implements the resulting action
        self.current_state = self.safe_exec(self.current_state, resulting_action)
        row, col = self.current_state  # just an alias

        # stores the next state
        sprime = self.current_state

        # retrieves the tile of current coordinates (' ', 'G' or 'H')
        tile = self.world[row][col]
        reward = self.rewards[tile]

        # terminal test (goal or hole)
        done = tile in 'GH'

        self.last_action = resulting_action
        info = {
            "a_idx": a,
            "a_name": self.action_names[a],
            "o_idx": o,
            "o_name": self.opponent_action_names[o],
            'r_idx': resulting_action,
            'r_name': self.action_names[resulting_action],
            "tile": '{}'.format(tile)
        }

        # makes the opponent learn now, minimaxQ receives both agent and opponent actions
        # (they are swapped and reward is negated because MinimaxQ is the opponent)
        if self.opp_name == 'MinimaxQ':
            self.opponent.learn(s, o, a, -reward, sprime, done)
        else:
            self.opponent.learn(s, o, -reward, sprime, done)

        # print(s, self.action_names[a], self.opponent_action_desc[o], -reward, sprime)

        return self.current_state, reward, done, info

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.current_state  # self.s // self.ncols, self.s % self.ncols
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.last_action is not None:
            outfile.write("  ({})\n".format(self.action_names[self.last_action]))
        else:
            outfile.write("\n")

        outfile.write("_" * (self.ncols + 2) + '\n')
        outfile.write("\n".join('|%s|' % ''.join(line) for line in desc) + '\n')
        outfile.write("‾" * (self.ncols + 2) + '\n\n')
        # ^possible issue with overline character (Unicode: U+203E)

        if mode != 'human':
            return outfile

    def _reset(self):
        self.current_state = self.initial_state()
        self.last_action = None
        return self.current_state

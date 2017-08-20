import gym
import gym.spaces
import numpy as np
import sys
from six import StringIO, b

from gym import utils

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "S   ",
        " H H",
        "   H",
        "H  G"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


class AdversarialGrid(gym.Env):
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

    # coordinate system is matrix-based (e.g. down increases the row)
    action_effects = {
        LEFT: (0, -1),
        DOWN: (+1, 0),
        RIGHT: (0, +1),
        UP: (-1, 0)
    }

    action_names = {
        LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up"
    }

    rewards = {
        'G': 1,
        'H': -1,
        'S': 0,
        ' ': 0
    }

    def __init__(self, desc=None, map_name="4x4"):
        self.current_state = (0, 0)

        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.world = desc

        # desc has the 'world' as an array
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrows, self.ncols = nrows, ncols = desc.shape

        self.number_actions = len(self.action_effects)  # number of actions
        self.number_states = nrows * ncols  # number of states

        self.action_space = gym.spaces.Discrete(self.number_actions)
        self.observation_space = gym.spaces.Discrete(self.number_states)

    def safe_exec(self, origin, a):
        """
        Simulates the execution of an action,
        preventing out of bounds
        :param origin: tuple(row, col)
        :param a:the action (an integer)
        :return:
        """
        row, col = origin
        # action effects are deterministic
        action_effect = self.action_effects[a]
        new_row, new_col = row + action_effect[0], col + action_effect[1]

        # ensures new coordinates are within boundaries
        new_row = min(self.nrows - 1, max(0, new_row))
        new_col = min(self.ncols - 1, max(0, new_col))

        return new_row, new_col

    def _step(self, a):
        """
        Receives the action of the agent and returns the outcome
        :param a:the action to execute (an integer)
        :return: tuple(state, reward, done, info)
        """
        self.current_state = self.safe_exec(self.current_state, a)
        row, col = self.current_state  # just an alias

        # retrieves the tile of current coordinates (' ', 'G' or 'H')
        tile = self.world[row][col]
        reward = self.rewards[tile]

        # terminal test (goal or hole)
        done = tile in 'GH'

        self.last_action = a
        info = {
            "action_index": a,
            "action_name": self.action_names[a],
            "tile": '{}'.format(tile)
        }
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
        outfile.write("\n".join(''.join(line) for line in desc) + "\n\n")

        if mode != 'human':
            return outfile

    def _reset(self):
        self.current_state = 0, 0
        self.last_action = None
        return self.current_state

import numpy as numpy
from copy import deepcopy
from gym.spaces.box import Box
import sys
sys.path.append('../')
from utils.helpers import Experience
class Env(object):
    def __init__(self, args, env_ind = 0):
        self.logger = args.logger
        self.ind = env_ind
        self.mode = args.mode
        self.seed = args.seed + self.ind
        self.visualize = args.visualize

        self.env_type = args.env_type
        self.game = args.game
        self._reset_experience()

    def _reset_experience(self):
        self.exp_state0 = None
        self.exp_action = None
        self.exp_reward = None
        self.exp_state1 = None
        self.exp_terminal1 = None
    def _get_experience(self):
        return Experience(state0 = self.exp_state0,
                          action = self.exp_action,
                          reward = self.exp_reward,
                          state1 = self._preprocessState(self.exp_state1),
                          terminal1 = self.exp_terminal1)
    def _preprocessState(self, state):
        raise NotImplementedError("not implemented in base class")
        
    @property
    def action_dim(self):
        if isinstance(self.env.action_space, Box):
            return self.env.action_space.shape[0]
        else:
            return self.env.action_space.n
import numpy as np
from collection import deque, namedtuple
import warnings
import random
import sys
sys.path.append("../")
from utils.helpers import Experience

from memory import sample_batch_indexs, zeroed_observation, RingBuffer, Memory

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)
        self.limit = limit
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs = None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexs(0, self.nb_entries -1, size = batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx-2] if idx >=2 else False
            while terminal0:
                idx = sample_batch_indexs(1, self.nb_entries, size = 1)[0]
                terminal0 = self.terminals[idx-2] if idx>=2 else False
            assert 1 <= idx < self.nb_entries

            state0 = [self.observations[idx-1]]
            for offset in range(0, self.window_length-1):
                current_idx = idx -2 -offset
                current_terminal = self.terminals[current_idx-1] if current_idx-1> 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    break
                state0.insert(0, self.observations[current_idx])
            while(len(state0) < self.window_length):
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx-1]
            reward = self.rewards[idx-1]
            terminal1 = self.terminals[idx-1]
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])
            experiences.append(Experience(state0, action , reward, state1, terminal1))
        return experiences
        
    @property
    def nb_entries(self):
        return len(self.observations)
    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config
import numpy as np
from agents import Agent
import torch
from torch.autograd import Variable
import time
class DQNAgent(Agent):
    def __init__(self, args, env_prototype, model_prototype, memory_prototype):
        super(DQNAgent, self).__init__(args, env_prototype, model_prototype, memory_prototype)
        self.logger.warning("<==================================>DQN")

        # env

        self.env = self.env_prototype(self.env_params)
        self.state_shape = self.env.state_shape
        self.action_dim = self.env.action_dim

        # model
        self.model_params.state_shape = self.state_shape
        self.model_params.action_dim = self.action_dim
        self.model = self.model_prototype(self.model_params)
        self._load_model(self.model_file)

        # target model
        self.target_model = self.model_prototype(self.model_params)
        self._update_target_model_hard()

        # memory
        self.memory_params = args.memory_params
        self._reset_states()
    def _reset_states(self):
        self._reset_experience()
        self.recent_action = None
        self.recent_observation = None
    def _reset_training_loggings(self):
        self.steps_avg_log = []
        self.steps_std_log = []
        self.reward_avg_log = []
        self.reward_std_log = []
        self.nepisodes_solved_log = []
        self.repisodes_solved_log = []
        if self.visualize:
            self.win_steps_avg = "win_steps_avg"
    def _update_target_model_hard(self):
        for i, (key, target_weights) in enumerate(self.target_model.state_dict().iteritems()):
            target_weights += self.target_model_update*self.model.state_dict[key]
    def fit_model(self):
        #memory
        self.memory = self.memory_prototype(limit = self.memory_params.memory_size, 
        window_length = self.memory_params.hist_len)
        self.eps = self.eps_start

        self.optimizer = self.optim(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        self.lr_adjusted = self.lr

        self.training = True
        self._reset_training_loggings()
        self.start_time = time.time()
        self.step = 0
        nepisodes = 0
        nepisodes_solved = 0
        episode_steps = None
        episode_reward = None
        total_reward = 0.
        should_start_new = True
        while self.step < self.steps:
            if should_start_new:
                episode_steps = 0
                episode_reward = 0.

                self._reset_states()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                if not self.training:
                    if self.visualize: self.env.visual()
                    if self.render: self.env.render()
                should_start_new = False
            
            action = self._forward(self.experience.state1)
            reward = 0.

            
        
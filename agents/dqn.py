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
            for _ in range(self.action_repetition):
                self.experience = self.env.step(action)
                if not self.training:
                    if self.visualize: self.env.visual()
                    if self.render: self.env.render()
                reward += self.experience.reward
                if self.experience.terminal1:
                    should_start_new = True
                    break
                if self.early_stop and (episode_steps + 1) >= self.early_stop or (self.step+1)%self.eval_freq == 0:
                    should_start_new = True
                if should_start_new:
                    self._backward(reward, True)
                else:
                    self._backward(reward, self.experience.terminal1)
                episode_steps += 1
                episode_reward += reward
                self.step += 1
                if should_start_new:
                    self._forward(self.experience.state1)
                    self._backward(0., False)
                    total_reward += episode_reward
                    nepisodes+=1
                    if self.experience.terminal1:
                        nepisodes_solved+=1
                    self._reset_states()
                    episode_reward = None
                    episode_steps = None
                if self.step % self.prog_freq == 0:
                    self.logger.warning("Reporting       @ Step: " + str(self.step) + " | Elapsed Time: " + str(time.time() - self.start_time))
                    self.logger.warning("Training Stats:   lr:               {}".format(self.lr_adjusted))
                    self.logger.warning("Training Stats:   epsilon:          {}".format(self.eps))
                    self.logger.warning("Training Stats:   total_reward:     {}".format(total_reward))
                    self.logger.warning("Training Stats:   avg_reward:       {}".format(total_reward/nepisodes if nepisodes > 0 else 0.))
                    self.logger.warning("Training Stats:   nepisodes:        {}".format(nepisodes))
                    self.logger.warning("Training Stats:   nepisodes_solved: {}".format(nepisodes_solved))
                    self.logger.warning("Training Stats:   repisodes_solved: {}".format(nepisodes_solved/nepisodes if nepisodes > 0 else 0.))

            # evaluation & checkpointing
                if self.step > self.learn_start and self.step % self.eval_freq == 0:
                    # Set states for evaluation
                    self.training = False
                    self.logger.warning("Evaluating      @ Step: " + str(self.step))
                    self._eval_model()

                    # Set states for resume training
                    self.training = True
                    self.logger.warning("Resume Training @ Step: " + str(self.step))
                    should_start_new = True
        def _eval_model(self):
            pass


        
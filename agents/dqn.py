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
    def _sample_validation_data(self):
        self.logger.warning("Validation Data @ Step: "+str(self.step))
        self.valid_data = self.memory.sample(self.valid_size)
    def _compute_validation_stats(self):
        return self._get_q_update(self.valid_data)
    def _get_q_update(self, experiences):
        state0_batch_vb = Variable(torch.from_numpy(np.array(tuple(experiences[i].state0 for i in range(len(experiences))))).type(self.dtype))
        action_batch_vb    = Variable(torch.from_numpy(np.array(tuple(experiences[i].action for i in range(len(experiences))))).long())
        reward_batch_vb    = Variable(torch.from_numpy(np.array(tuple(experiences[i].reward for i in range(len(experiences)))))).type(self.dtype)
        state1_batch_vb    = Variable(torch.from_numpy(np.array(tuple(experiences[i].state1 for i in range(len(experiences))))).type(self.dtype))
        terminal1_batch_vb = Variable(torch.from_numpy(np.array(tuple(0. if experiences[i].terminal1 else 1. for i in range(len(experiences)))))).type(self.dtype)
        if self.use_cuda:
            action_batch_vb = action_batch_vb.cuda()
        if self.enable_double_dqn:
            q_values_vb = self.model(state1_batch_vb)
            q_values_vb = Variable(q_values_vb.data)
            _, q_max_action_vb = q_values_vb.max(dim = 1, keepdim = True)
            next_max_q_values_vb = self.target_model(state1_batch_vb)
            next_max_q_values_vb = Variable(next_max_q_values_vb.data)
            next_max_q_values_vb = next_max_q_values_vb.gather(1, q_max_action_vb)
        else:
            next_max_q_values_vb = self.target_model(state1_batch_vb)
            next_max_q_values_vb = Variable(next_max_q_values_vb)
            next_max_q_values_vb, _  = next_max_q_values_vb.max(dim = 1, keepdim = True)
        current_q_values_vb = self.model(state0_batch_vb).gather(1, action_batch_vb.unsqueeze(1)).squeeze()
        next_max_q_values_vb = next_max_q_values_vb*terminal1_batch_vb.unsqueeze(1)
        expected_q_values_vb = reward_batch_vb + self.gamma*next_max_q_values_vb.squeeze()
        td_error_vb = self.value_criteria(current_q_values_vb, expected_q_values_vb)
        if not self.training:
            td_error_vb = Variable(td_error_vb.data)
        return next_max_q_values_vb.clone().mean(), td_error_vb
    def _epsilon_greedy(self, q_values_ts):
        pass
    def _forward(self, observation):
        state = self.memory.get_recent_state(observation)
        state_ts = torch.from_numpy(np.array(state)).unsqueeze(0).type(self.dtype)
        q_value_ts = self.model(Variable(state_ts, volatile = True)).data
        if self.training and self.step< self.learn_start:
            action = random.randrange(self.action_dim)
        else:
            action = self._epsilon_greddy(q_value_ts)
        self.recent_observation = observation
        self.recent_action = action 
        return action
    def _backward(self, reward, terminal):
        if self.step % self.memory_interval == 0:
            self.memory.append(self, self.recent_observation, self.recent_action, reward, terminal,
            training =  self.training)
        if not self.training:
            return
        if self.step == self.learn_start +1:
            self._sample_validation_data()
            self.logger.warning(" Statrt Training @ Step: "+ str(self.step))
        if self.step > self.learn_start and self.step%self.train_interval==0:
            experiences = self.memory.sample(self.batch_size)
            _, td_error_vb = self._get_q_update(experiences)
            self.optimizer.zero_grad()
            td_error_vb.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-self.clip_grad, self.clip_grad)
            self.optimizer.step()
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
                self.training = True
                self.logger.warning("Resume Training @ Step: " + str(self.step))
                should_start_new = True
        def _eval_model(self):
            pass


        
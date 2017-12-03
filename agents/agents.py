import torch
import torch.optim as optim
import sys
sys.path.append('../')
from utils.helpers import Experience
class Agent(object):
    def __init__(self, args, env_prototype, model_prototype, memory_prototype,):
        self.logger = args.logger
        self.env_prototype = env_prototype
        self.env_params = args.env_params
        self.model_params = args.model_params
        self.model_prototype = model_prototype
        self.memory_prototype = memory_prototype
        self.memory_params = args.memory_params

        # params
        self.model_name = args.model_name
        self.model_file = args.model_file

        self.render = args.render
        self.visualize = args.visualize
        self.save_best = args.save_best
        if self.save_best:
            self.best_step = None
            self.best_reward = None
        self.hist_len = args.hist_len
        self.hidden_dim = args.hidden_dim
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        ##agent params
        self.value_criteria = args.value_criteria
        self.optim = args.optim

        ## hyper parameters
        self.steps = args.steps
        self.early_stop = args.early_stop
        self.gamma = args.gamma
        self.clip_grad = args.clip_grad
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.eval_freq = args.eval_freq
        self.eval_steps = args.eval_steps
        self.prog_freq = args.prog_freq
        self.test_nepisodes = args.test_nepisodes
        if args.agent_type == 'dqn':
            self.enable_double_dqn = args.enable_double_dqn
            self.enable_dueling = args.enable_dueling
            self.dueling_type = args.dueling_type

            self.learn_start = args.learn_start
            self.batch_size = args.batch_size
            self.valid_size = args.valid_size
            self.eps_start = args.eps_start
            self.eps_end = args.eps_end
            self.eps_eval = args.eps_eval
            self.eps_decay = args.eps_decay
            self.target_model_updata = args.target_model_updata
            self.action_repetition = args.action_repetition
            self.memory_interval = args.memory_interval
            self.train_interval = args.train_interval
    def _reset_experience(self):
        self.experience = Experience(state0 = None,
        action = None,
        reward = None,
        state1 = None, 
        terminal1 = None)
    def _load_model(self, model_file):
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
        else:
            self.logger.warning("No pretrained Model. Will Train From Scratch")
    def _save_model(self, step, curr_reward):
        if self.save_best:
            if self.best_step is None:
                self.best_step = step
                self.best_reward = curr_reward
            if curr_reward >=self.best_reward:
                self.best_step = step
                self.best_reward = curr_reward
                torch.save(self.model.state_dict(), self.model_name)
        else:
            torch.save(self.model.state_dict(), self.model_file)
    def _forward(self, observation):
        raise NotImplementedError
    def _backward(self, reward, terminal):
        raise NotImplementedError
    def _eval_model(self):
        raise NotImplementedError
    def fit_model(self):
        raise NotImplementedError
    def test_model(self):
        raise NotImplementedError
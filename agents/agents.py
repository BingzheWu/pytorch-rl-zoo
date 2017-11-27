import torch
import torch.optim as optim

class Agent(object):
    def __init__(self, args, env, model,):
        self.logger = args.logger
        self.env = env

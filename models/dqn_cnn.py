import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.init_weights import init_weights, normalized_columns_initializer

from model import Model

class DQNCnnModel(Model):
    def __init__(self, args):
        super(DQNCnnModel, self).__init__(args)

        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size = 3, stride = 2)
        self.rl1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2)
        self.rl2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2)
        self.rl3 = nn.ReLU()
        self.fc4 = nn.Linear(32*5*5, self.hidden_dim)
        self.rl4 = nn.ReLU()
        if self.enable_dueling:
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims+1)
            self.v_ind = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0)
            self.a_ind = torch.LongTensor(np.arange(1, self.output_dims+1)).unsqueeze(0)
        else:
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()
    def _init_weights(self):
        self.apply(init_weights)
        self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.0001)
        self.fc4.bias.data.fill_(0)
        self.fc5.weight.data = normalized_columns_initializer(self.fc5.weight.data, 0.0001)
        self.fc5.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims[0], self.input_dims[1], self.input_dims[1])
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = self.rl3(self.conv3(x))
        x = self.rl4(self.fc4(x.view(x.size(0), -1)))
        if self.enable_dueling:
            x = self.fc5(x)
            v_ind_vb = Variable(self.v_ind)
            a_ind_vb = Variable(self.a_ind)
            if self.use_cuda:
                v_ind_vb = v_ind_vb.cuda()
                a_ind_vb = a_ind_vb.cuda()
            v = x.gather(1, v_ind_vb.expand(x.size(0), self.output_dims))
            a = x.gather(1, a_ind_vb.expand(x.size(0), self.output_dims))
            # now calculate Q(s, a)
            if self.dueling_type == "avg":      # Q(s,a)=V(s)+(A(s,a)-avg_a(A(s,a)))
                x = v + (a - a.mean(1).expand(x.size(0), self.output_dims))
            elif self.dueling_type == "max":    # Q(s,a)=V(s)+(A(s,a)-max_a(A(s,a)))
                x = v + (a - a.max(1)[0].expand(x.size(0), self.output_dims))
            elif self.dueling_type == "naive":  # Q(s,a)=V(s)+ A(s,a)
                x = v + a
            else:
                assert False, "dueling_type must be one of {'avg', 'max', 'naive'}"
            del v_ind_vb, a_ind_vb, v, a
            return x
        else:
            return self.fc5(x.view(x.size(0), -1))

    
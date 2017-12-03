from env import Env
import gym
import sys
import random
sys.path.append('../')
from utils.options import EnvParams
class GymEnv(Env):
    def __init__(self, args, env_ind = 0):
        super(GymEnv, self).__init__(args, env_ind)
        assert self.env_type == "gym"
        self.env = gym.make(self.game)
        self.env.seed(self.seed)
        # action setup
        self.actions = range(self.action_dim)
        
        if args.agent_type == "a3c":
            self.enable_continuous = args.enable_continuous
        else:
            self.enable_continuous = False
    def _preprocessState(self, state):
        return state

    @property
    def state_shape(self):
        return self.env.observation_space.shape[0]
    def render(self):
        if self.mode == 2:
            frame = self.env.render(mode = 'rgb_array')
            frame_name = self.img_dir + "frame_%04d.jpg" % self.frame_ind
            self.imsave(frame_name, frame)
            slef.frame_ind += 1
            return frame
        else:
            return self.env.render()
    def reset(self):
        self._reset_experience()
        self.exp_state1 = self.env.reset()
        return self._get_experience()
    def step(self, action_index):
        self.exp_action = action_index
        if self.enable_continuous:
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.exp_action)
        else:
            self.exp_state1, self.exp_reward, self.exp_terminal1,_ = self.env.step(self.actions[self.exp_action])
        return self._get_experience()

def test():
    args = EnvParams()
    print(args.game)
    print(args.env_type)
    test_env = GymEnv(args)
    print(test_env.mode)
    test_env.env.reset()
    for _ in range(1000):
        test_env.render()
        action_idx = random.randint(0,2)
        x = test_env.step(action_idx)
        print(type(x))
if __name__ == '__main__':
    test()
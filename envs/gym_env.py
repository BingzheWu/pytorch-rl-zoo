from env import Env
import gym
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
            pass
        else:
            return self.env.render()
    def step(self, action_index):
        self.exp_action = action_index
        if self.enable_continuous:
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.exp_action)
        else:
            self.exp_state1, self.exp_reward, self.exp_terminal1,_ = self.env.step(self.actions[self.exp_action])
        return self._get_experience()
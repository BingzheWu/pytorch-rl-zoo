from envs.gym import GymEnv
EnvDict = {"gym":       GymEnv}

#from models.empty import EmptyModel
#from models.dqn_mlp import DQNMlpModel
from models.dqn_cnn import DQNCnnModel
#from models.a3c_mlp_con import A3CMlpConModel
#from models.a3c_cnn_dis import A3CCnnDisModel
#from models.acer_mlp_dis import ACERMlpDisModel
#from models.acer_cnn_dis import ACERCnnDisModel
ModelDict = {"dqn-cnn":      DQNCnnModel,}

from memories.sequential import SequentialMemory
from memories.episode_parameter import EpisodeParameterMemory
from memories.episodic import EpisodicMemory
MemoryDict = {"sequential":        SequentialMemory,        # off-policy
              "none":              None}                    # on-policy

#from core.agents.empty import EmptyAgent
from core.agents.dqn   import DQNAgent
#from core.agents.a3c   import A3CAgent
#from core.agents.acer  import ACERAgent
AgentDict = {              # to test integration of new envs, contains only the most basic control loop
             "dqn":   DQNAgent,                 # dqn  (w/ double dqn & dueling as options
             }              

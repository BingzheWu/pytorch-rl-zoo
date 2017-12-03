import numpy as np
from utils.options import Options
from utils.factory import EnvDict, ModelDict, MemoryDict, AgentDict
opt = Options()
np.random.seed(opt.seed)

env_prototype = EnvDict[opt.env_type]
memory_prototype = MemoryDict[opt.memory_type]
model_prototype = ModelDict[opt.model_type]
agent = AgentDict[opt.agent_type](opt.agent_params,
env_prototype = env_prototype,
model_prototype = model_prototype,
memory_prototype = memory_prototype)

if opt.mode == 1:
    agent.fit_model()
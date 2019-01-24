#import numpy as np

#from curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv
#env = Arm3dDiscEnv()

#from baselines.trpo_mpi import trpo_mpi

#import baselines.common.vec_env
#env = vec_env.vec_normalize(env)

#trpo_mpi.learn(network='mlp', env=env, total_timesteps=2)


#os = env.observation_space
#print(type(os))
#os = env.observation_space
#print(type(os))
#a = env.action_space
#print(type(a))
#a = env.action_space
#print(type(a))

import numpy as np
# from curriculum.envs.maze.point_env import PointEnv
from env_wrapper_rllab_to_openai import WrappedPointMazeEnv
env = WrappedPointMazeEnv()
env.reset()

for i in range(250):
    env.step(np.array([0.05, 0.05]))
    env.render()

from baselines.ppo2 import ppo2
# env_wrapper_rllab_to_openai
model = ppo2.learn(network='mlp', env=env, total_timesteps=2)

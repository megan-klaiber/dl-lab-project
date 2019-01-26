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



# from curriculum.envs.maze.point_env import PointEnv
#for i in range(250):
    # env.step(np.array([0.05, 0.05]))
    # env.render()

import time
import numpy as np

from baselines.ppo2 import ppo2
from env_wrapper_rllab_to_openai import WrappedPointMazeEnv


env = WrappedPointMazeEnv()
env.reset()

model = ppo2.learn(network='mlp', env=env, total_timesteps=8, nsteps=8)

for i in range(250):
    obs = env.get_current_obs(); print("obs: ", obs)
    s = model.step(obs); print("s: ", s)
    env.step(s[0])
    env.render()
    time.sleep(2)

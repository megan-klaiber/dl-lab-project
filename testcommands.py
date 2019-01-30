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
import tensorflow as tf

from baselines.ppo2 import ppo2
from env_wrapper_rllab_to_openai import WrappedPointMazeEnv


env = WrappedPointMazeEnv()
# env.post_init_stuff(max_env_timestep=150, fixed_restart_state=np.array([0.8, 0]))
env.post_init_stuff(eval_runs=2, max_env_timestep=500, do_rendering=True, sampling_method='sample_nearby',
                    steps_per_curriculum = 10000)
# Idee:
# max_env_timestep=500
# nsteps = 50000
# total_timesteps = 300 * nsteps

model = ppo2.learn(network='mlp',
                   env=env,
                   save_interval=100,
                   total_timesteps=300 * 50000,
                   nsteps=50000,
                   nminibatches=1,
                   num_layers=2,
                   num_hidden=64,
                   activation=tf.nn.relu)
env.save()

#model = ppo2.learn(network='mlp', env=env, total_timesteps=100000, nsteps=1, nminibatches=1,
                   #num_layers=2,
                   #num_hidden=64,
                   #activation=tf.nn.relu)

#env.reset()
#for i in range(250):
    #obs = env.get_current_obs(); print("obs: ", obs)
    #s = model.step(obs); print("s: ", s)
    #env.step(s[0])
    #env.render()
    #time.sleep(2)

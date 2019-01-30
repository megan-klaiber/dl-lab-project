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
import os

from baselines.ppo2 import ppo2
from env_wrapper_rllab_to_openai import WrappedPointMazeEnv


env = WrappedPointMazeEnv()
# Final configuration parameters:
# eval_runs = 10
# max_env_timestep = 500
# do_rendering = False
# sampling_method='sample_nearby'
# steps_per_curriculum = 50000
# nsteps = steps_per_curriculum
# total_timesteps = 300 * nsteps
# save_interval = 100
# verbose = False


# Parameters for testing for short runs:
eval_runs = 2
max_env_timestep = 150
do_rendering = True
sampling_method = 'uniform'
steps_per_curriculum = 10000
nsteps = steps_per_curriculum
total_timesteps = 3 * nsteps
save_interval = 2
verbose = True

env.post_init_stuff(eval_runs=eval_runs, max_env_timestep=max_env_timestep, do_rendering=do_rendering,
                    sampling_method=sampling_method,
                    steps_per_curriculum=steps_per_curriculum,
                    verbose=verbose)

model = ppo2.learn(network='mlp',
                   env=env,
                   save_interval=save_interval,
                   total_timesteps=total_timesteps,
                   nsteps=nsteps,
                   nminibatches=1,
                   num_layers=2,
                   num_hidden=64,
                   activation=tf.nn.relu)
env.save()
# Save the final state of the trained model.
model_dir_path = os.path.join("results", "model")
model_file_path = os.path.join(model_dir_path, "final_trained_model")
print('Saving model to: ', model_file_path)
os.makedirs(model_dir_path, exist_ok=True)
model.save(model_file_path)

# Only rendering:
# env.reset()
# for i in range(250):
#      env.render()

# env.reset()
# for i in range(250):
#     obs = env.get_current_obs(); print("obs: ", obs)
#     s = model.step(obs); print("s: ", s)
#     env.step(s[0])
#     env.render()
#     time.sleep(2)

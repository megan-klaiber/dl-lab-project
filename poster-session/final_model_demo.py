import sys

sys.path.append("../dl-lab-project/")

from baselines.ppo2 import ppo2
from env_wrapper_rllab_to_openai import WrappedPointMazeEnv
import tensorflow as tf

import time
import numpy as np
import random

def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

outer_iter = 0
# get all arguments
eval_runs = 30
max_env_timestep = 1000
do_rendering = True
sampling_method = "uniform"
steps_per_curriculum = 1
nsteps = 1
total_timesteps = outer_iter * steps_per_curriculum
save_interval = 0
verbose = True
seed = 42
sample_on_goal_area = False

model_path = "../dl-lab-project/poster-session/model/model_2019-02-03_12-56-21_all_previous_40"

# initialize environment
env = WrappedPointMazeEnv()
env.post_init(eval_runs=eval_runs, max_env_timestep=max_env_timestep, do_rendering=do_rendering,
              sampling_method=sampling_method,
              steps_per_curriculum=steps_per_curriculum,
              verbose=verbose,
              sample_on_goal_area=sample_on_goal_area,
              model_file_path="temp",
              eval_starts_file_name="temp_eval",
              eval_results_file_name="temp_eval_results")

              # train model
model = ppo2.learn(network='mlp',
                   load_path=model_path,
                   env=env,
                   save_interval=save_interval,
                   total_timesteps=total_timesteps,
                   nsteps=nsteps,
                   nminibatches=1,
                   num_layers=2,
                   num_hidden=64,
                   activation=tf.nn.relu,
                   gamma=0.998,
                   lr=0.01,
                   seed=seed)

for i in range(100):
    time.sleep(0.2)
    env.render()

env.set_model(model)
np.random.seed(1)
env.evaluate()

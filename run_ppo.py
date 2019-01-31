import time
import datetime
import numpy as np
import tensorflow as tf
import os
import random
import argparse

from baselines.ppo2 import ppo2
from env_wrapper_rllab_to_openai import WrappedPointMazeEnv

# Final configuration parameters:
# eval_runs = 10
# max_env_timestep = 500
# do_rendering = False
# sampling_method='sample_nearby'
# steps_per_curriculum = 50000
# nsteps = 50000
# total_timesteps = 400 * nsteps
# save_interval = 0
# verbose = False
# sample_on_goal_area = True

# Parameters for testing for short runs:
# eval_runs = 2
# max_env_timestep = 150
# do_rendering = True
# sampling_method = 'uniform'
# steps_per_curriculum = 1500
# nsteps = steps_per_curriculum
# total_timesteps = 3 * nsteps
# save_interval = 2
# verbose = True
# sample_on_goal_area = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_runs', default=10, type=int, help='How many runs for evaluation during training')
    parser.add_argument('--max_env_timestep', default=500, type=int, help='')
    parser.add_argument('--do_rendering', default=False, type=bool, help='')
    parser.add_argument('--sampling_method', default='uniform', type=str, help='')
    parser.add_argument('--steps_per_curriculum', default=50000, type=int, help='')
    parser.add_argument('--nsteps', default=50000, type=int, help='')
    parser.add_argument('--total_timesteps', default=400, type=int, help='outer iters / number multiply with nsteps')
    parser.add_argument('--save_interval', default=0, type=int, help='')
    parser.add_argument('--verbose', default=False, type=bool, help='print more information')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--sample_on_goal_area', default=True, type=bool, help='')
    args = parser.parse_args()

    # get all arguments
    eval_runs = args.eval_runs
    max_env_timestep = args.max_env_timestep
    do_rendering = args.do_rendering
    sampling_method = args.sampling_method
    steps_per_curriculum = args.steps_per_curriculum
    nsteps = args.nsteps
    total_timesteps = args.total_timesteps * nsteps
    save_interval = args.save_interval
    verbose = args.verbose
    seed = args.seed
    sample_on_goal_area = args.sample_on_goal_area

    # set random seed
    random.seed(seed)
    np.random.seed(seed)

    # initialize environment
    env = WrappedPointMazeEnv()
    env.post_init_stuff(eval_runs=eval_runs, max_env_timestep=max_env_timestep, do_rendering=do_rendering,
                        sampling_method=sampling_method,
                        steps_per_curriculum=steps_per_curriculum,
                        verbose=verbose,
                        sample_on_goal_area=sample_on_goal_area)

    # train model
    model = ppo2.learn(network='mlp',
                       env=env,
                       save_interval=save_interval,
                       total_timesteps=total_timesteps,
                       nsteps=nsteps,
                       nminibatches=1,
                       num_layers=2,
                       num_hidden=64,
                       activation=tf.nn.relu)

    # create results directory
    if not os.path.exists("results"):
        os.mkdir("results")
    # create experiment directory
    # get datetime
    experiment_date = datetime.datetime.today().strftime(
        "%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join('results',"ppo_maze_{}_{}_{}".format(experiment_date, sampling_method, seed))
    os.mkdir(experiment_dir)

    # to use evaluation see instructions in env_wrapper_rllab_to_openai.evaluate
    # save evaluation results
    eval_file_name = os.path.join(experiment_dir, 'evaluation_{}_{}_{}.json'.format(experiment_date, sampling_method, seed))
    env.save(file_name=eval_file_name)

    # Save the final state of the trained model.
    #model_dir_path = os.path.join("results", "model")
    #model_file_path = os.path.join(model_dir_path, "final_trained_model")
    # os.makedirs(model_dir_path, exist_ok=True)
    os.mkdir(os.path.join(experiment_dir, "model"))
    model_file_path = os.path.join(experiment_dir, 'model' ,'final_trained_model_{}_{}_{}'.format(experiment_date, sampling_method, seed))
    print('Saving model to: ', model_file_path)
    model.save(model_file_path)

import argparse
import datetime
import os
import random
import time

import numpy as np
import tensorflow as tf

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
    parser.add_argument('--eval_runs', default=50, type=int, help='How many runs for evaluation during training.')
    parser.add_argument('--max_env_timestep', default=500, type=int, help='Maximum number of timesteps taken in an environment.')
    parser.add_argument('--do_rendering', default=False, action='store_true', help='True for render simulations.')
    parser.add_argument('--sampling_method', default='uniform', type=str,
                        help='Defines the samppling method. Can be "uniform"')
    parser.add_argument('--steps_per_curriculum', default=50000, type=int, help='')
    parser.add_argument('--nsteps', default=50000, type=int, help='')
    parser.add_argument('--outer_iter', default=400, type=int, help='outer iters')
    parser.add_argument('--save_interval', default=0, type=int, help='')
    parser.add_argument('--verbose', default=False, action='store_true', help='print more information')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--sample_on_goal_area', default=False, action='store_true', help='')
    args = parser.parse_args()

    # get all arguments
    eval_runs = args.eval_runs
    max_env_timestep = args.max_env_timestep
    do_rendering = args.do_rendering
    sampling_method = args.sampling_method
    steps_per_curriculum = args.steps_per_curriculum
    nsteps = args.nsteps
    total_timesteps = args.outer_iter * steps_per_curriculum
    save_interval = args.save_interval
    verbose = args.verbose
    seed = args.seed
    sample_on_goal_area = args.sample_on_goal_area

    param_info = ("\nRunning 'run_ppo.py' with following parameters: \n"
                  "\teval_runs: {}\n"
                  "\tmax_env_timestep: {}\n"
                  "\tdo_rendering: {}\n"
                  "\tsampling_method: {}\n"
                  "\tsteps_per_curriculum: {}\n"
                  "\tnsteps: {}\n"
                  "\touter_iter: {}\n"
                  "\ttotal_timesteps: {} (= outer_iter * steps_per_curriculum)\n"
                  "\tsave_interval: {}\n"
                  "\tverbose: {}\n"
                  "\tseed: {}\n"
                  "\tsample_on_goal_area: {}\n"
                  "".format(eval_runs, max_env_timestep, do_rendering, sampling_method,
                            steps_per_curriculum, nsteps, args.outer_iter,
                            total_timesteps, save_interval,
                            verbose, seed, sample_on_goal_area)
                  )

    print(param_info)

    # set random seed
    random.seed(seed)
    np.random.seed(seed)

    # create results directory
    if not os.path.exists("results"):
        os.mkdir("results")
    # create experiment directory
    # get datetime
    experiment_date = datetime.datetime.today().strftime(
        "%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join('results',"ppo_maze_{}_{}_{}".format(experiment_date, sampling_method, seed))
    os.mkdir(experiment_dir)
    
    os.mkdir(os.path.join(experiment_dir, "model"))
    model_file_path = os.path.join(experiment_dir, 'model' ,'model_{}_{}_{}'.format(experiment_date, sampling_method, seed))

    # to use evaluation see instructions in env_wrapper_rllab_to_openai.evaluate
    # save evaluation results
    eval_starts_file_name = os.path.join(experiment_dir, 'evaluation_starts_{}_{}_{}.json'.format(experiment_date, sampling_method, seed))
    eval_results_file_name = os.path.join(experiment_dir, 'evaluation_results_{}_{}_{}.json'.format(experiment_date, sampling_method, seed))

    config_filename = os.path.join(experiment_dir, 'config_{}_{}_{}.txt'.format(experiment_date, sampling_method, seed))
    with open(config_filename, "w") as config_file:
        config_file.write(param_info)
    
    # initialize environment
    env = WrappedPointMazeEnv()
    env.post_init(eval_runs=eval_runs, max_env_timestep=max_env_timestep, do_rendering=do_rendering,
                  sampling_method=sampling_method,
                  steps_per_curriculum=steps_per_curriculum,
                  verbose=verbose,
                  sample_on_goal_area=sample_on_goal_area,
                  model_file_path=model_file_path,
                  eval_starts_file_name=eval_starts_file_name,
                  eval_results_file_name=eval_results_file_name)

    # train model
    model = ppo2.learn(network='mlp',
                       env=env,
                       save_interval=save_interval,
                       total_timesteps=total_timesteps,
                       nsteps=nsteps,
                       nminibatches=1,
                       num_layers=2,
                       num_hidden=64,
                       activation=tf.nn.relu,
                       gamma=0.99,
                       lr=0.01)

    # Last evaluation run, including saving the evaluation results and the model.
    env.evaluate()


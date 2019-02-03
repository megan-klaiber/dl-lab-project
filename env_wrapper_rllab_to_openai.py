import json
import os
import random
import time

import numpy as np

from curriculum.envs.maze.point_maze_env import PointMazeEnv
from gym.spaces import Box


class WrappedPointMazeEnv(PointMazeEnv):
    ''' Wrapper class for the PointMazeEnv of rllab to make it usable
        with OpenAI baselines PPO2 algorithm.

        For this the following components have been added:
        - establish general compatibility between rllab environment and PPO2 algorithm 
          (e.g. return expected data types to PPO2 algorithm)
        - handle model evaluation in the environment
        - handle curriculum generation/adaptation
    '''

    def __init__(self):
        # goal radius as given in the paper
        self.goal_radius = 0.3
        super().__init__(coef_inner_rew=1.0, maze_id=11, reward_dist_threshold=self.goal_radius)
        self.num_envs = 1     # required py PPO2
        self.episodes_steps = []
        self.episodes_goal_reached = []
        # integer that saves the index of the current start during training
        self.current_start = None
        # initializing parameters
        self.global_train_steps = 0
        self.eval_starts = {}
        self.eval_results = {}
        self.done_in_previous_step = False
        # range of good_starts
        self.R_max = 0.9
        self.R_min = 0.1
        # set goal
        self.goal = (4, 4)
        result = super().reset(goal=self.goal)
        # visualize goal radius
        tmp = np.copy(self.wrapped_env.model.geom_size)
        tmp[-1,0] = self.goal_radius * 0.9
        self.wrapped_env.model.geom_size = tmp

    def post_init(self, max_env_timestep, eval_runs, sampling_method, do_rendering,
                steps_per_curriculum, verbose, sample_on_goal_area,
                model_file_path, eval_starts_file_name, eval_results_file_name):
        ''' Further initialization of attributes because initialization in init method was
            not possible due to parameter handling in parent class. 

            Params:
            -------
            model_file_path: str
                             Path where to save the model
            eval_starts_file_name: str
                                   Path where to save the starts used during evaluation
            eval_results_file_name: str
                                    Path where to save the evaluation results 

            Other params explained in run_ppo.py
        '''
        self.max_env_timestep = max_env_timestep
        self.eval_runs = eval_runs
        self.sampling_method = sampling_method # can be either 'good_starts', 'all_previous' or 'uniform'
        self.do_rendering = do_rendering
        self.steps_per_curriculum = steps_per_curriculum
        self.verbose = verbose
        self.sample_on_goal_area = sample_on_goal_area
        self.model_file_path = model_file_path
        self.eval_starts_file_name = eval_starts_file_name
        self.eval_results_file_name = eval_results_file_name

        if self.sampling_method not in ['uniform', 'good_starts', 'all_previous']:
            raise ValueError("Unknown sampling method.")

        if self.sampling_method != 'uniform':
            # intialize start state list with states near goal
            if self.sample_on_goal_area:
                part_of_M = 100
                length = np.random.uniform(0, self.goal_radius, part_of_M)
                angle = np.pi * np.random.uniform(0, 2, part_of_M)
                x = self.wrapped_env.current_goal[0] + length * np.cos(angle)
                y = self.wrapped_env.current_goal[1] + length * np.sin(angle)
                starts_in_goal_area = list(zip(x, y))
                if self.verbose:
                    print("starts_in_goal_area:\n", starts_in_goal_area)
                self.curriculum_starts = self.sample_nearby(starts_in_goal_area)
            else:
                # first initialize curriculum_starts with states close to the goal state
                self.curriculum_starts = self.sample_nearby([self.wrapped_env.current_goal])
            if self.verbose:
                print("self.curriculum_starts: ", self.curriculum_starts)
            # intialize curriculum parameters
            self.all_starts = self.curriculum_starts
            self.start_counts = np.zeros(self.curriculum_starts.shape[0])
            self.goal_counts = np.zeros(self.curriculum_starts.shape[0])

    @property
    def observation_space(self):
        ''' Wrapper function for returning values using data types accepted by PPO2. '''
        ob_space = super().observation_space
        return Box(low=ob_space.low, high=ob_space.high, dtype=np.float32)

    @property
    def action_space(self):
        ''' Wrapper function for returning values using data types accepted by PPO2. '''
        ac_space = super().action_space
        return Box(low=ac_space.low, high=ac_space.high, dtype=np.float32)

    def step(self, actions):
        ''' Wrapper function for returning values using data types accepted by PPO2, also
            handles curriculum adaptations and calls evaluate function.
        '''
        # check if evaluation should be executed and print outer iteration
        if (self.global_train_steps % self.steps_per_curriculum == 0):
            print("self.global_train_steps / self.steps_per_curriculum: {0}".format(
                            self.global_train_steps / self.steps_per_curriculum))
            self.evaluate()
        self.global_train_steps += 1

        # check if environment should be reset to new start state
        if (self.done_in_previous_step or (self.episodes_steps[-1] >= self.max_env_timestep)) or \
            ((self.global_train_steps % self.steps_per_curriculum == 0) and self.sampling_method != 'uniform'):

            # check if goal reached 
            if self.done_in_previous_step:
                self.episodes_goal_reached.append(True)
                if self.sampling_method != 'uniform':
                    self.goal_counts[self.current_start] += 1
            else:
                self.episodes_goal_reached.append(False)
                
            # check if curriculum should be updated
            if (self.global_train_steps % self.steps_per_curriculum == 0) \
                and self.sampling_method != 'uniform':
                # find good starts
                # if we want to keep the starts that have not been used before,
                # change out to np.ones_like(self.goal_counts) * 0.5
                goal_reach_frequencies = np.divide(self.goal_counts,
                                                   self.start_counts,
                                                   out=np.ones_like(self.goal_counts) * np.NaN,
                                                   where=self.start_counts!=0)
                if self.verbose:
                    print('Number of starts sampled from:', len(goal_reach_frequencies))
                    print('Number of times it was sampled: ', np.sum(self.start_counts))
                    print('Frequencies with which goal is reached:')
                    print(goal_reach_frequencies)
                # check if only good starts or all previous starts used for curriculum update
                if self.sampling_method == 'all_previous':
                    starts = self.curriculum_starts
                else:
                    # line 5 in for loop of algorithm 1 in reverse curriculum paper
                    starts = self.good_starts(self.curriculum_starts, goal_reach_frequencies)
                # line 6 (last line) in for loop of algorithm 1 in reverse curriculum paper
                self.all_starts = np.concatenate((self.all_starts,starts))
                # first line in for loop of algorithm 1 in reverse curriculum paper
                if self.verbose:
                    print('Now sampling new starts...')
                # sample new start states with brownian motion
                self.curriculum_starts = self.sample_nearby(starts)
                if self.verbose:
                    print('...finished sampling new starts')
                    print('Sampled new starts: ', self.curriculum_starts)
                # line 2 in for loop of algorithm 1 in reverse curriculum paper
                self.curriculum_starts = np.concatenate((self.curriculum_starts,
                                                            self.sample_n(self.all_starts, 100)))

                # reset the counts of how often a start state is used and how often the goal is reached
                self.start_counts = np.zeros(self.curriculum_starts.shape[0])
                self.goal_counts = np.zeros(self.curriculum_starts.shape[0])

            # reset the environment to new start state from curriculum
            obs = self.reset(evaluate=False) 
            self.done_in_previous_step = False
            # prepare output values of the method
            rewards = np.array([0.0])
            dones = np.array([False])

        else:
            # perform step
            obs, rewards, dones, _ = super().step(actions)
            self.episodes_steps[-1] += 1
            
            if dones:
                self.done_in_previous_step = True
            # prepare output values of the method
            rewards = np.array([rewards])    # Helps with command 'ev = explained_variance(values, returns)' in ppo2.py
            dones = np.array([dones])    # Caused problems in runner.py at the end, when applying "sf01(mb_dones)".
            
        if self.do_rendering:
            self.render()
                
        infos = {}   # Caused problems in runner.py, with part "for info in infos: ..."
        return obs, rewards, dones, infos

    def good_starts(self, states, success_freq):
        ''' Determines the good starts of param "states" through success frequencies given 
            in "success_freq".
        '''
        starts = np.array([states[i] for i in range(len(states))
                        if (success_freq[i] > 0.1 and success_freq[i] < 0.9)])
        if self.verbose:
            print('Number of good starts', len(starts))
            print('Good starts: ', starts)
        if len(starts) == 0:
            easy_starts = np.array([states[i] for i in range(len(states))
                            if success_freq[i] >= 0.9])
            starts = easy_starts
            if len(easy_starts) == 0:
                if self.verbose:
                    print('No good or easy starts. Using starts from last iteration again.')
                return states
        return starts

    def sample_nearby(self, states, n_new=200, variance=1.0, t_b=50, M=1000):
        ''' Performs brownian motion from sampled states in param "states" and
            returns a subsample of size "n_new" of these.

            Params:
            -------
            states: numpy.ndarray
                    array of start states
            n_new: int
                   number of sampled starts in the ned
            variance: float
                      variance of normal distribution used for brownian motion
            t_b: int
                 horizon of one trajectory from a sampled start state
            M: int
               number of state list after adding starts
        '''
        starts = np.array(states)
        while(len(starts) < M):
            # sample a start state
            s_0 = random.choice(starts)
            # reset the env to the sampled start state
            super().reset(init_state=s_0, goal=self.goal)
            for i in range(t_b):
                # sample a random action
                a = np.random.normal(scale=variance, size=self.action_dim)
                # perform the random step
                _obs, _rew, _done, _env_info = super().step(a)
                # only want the start state part of the current state
                s_1 = self.get_current_obs()[:2].reshape(1, -1)
                # check if this results in a goal state
                if not self.is_in_goal_area(self.get_current_obs()):
                    starts = np.append(starts, s_1, axis=0)
                if self.do_rendering:
                    self.render()

        # now sample from the M start states
        new_starts = self.sample_n(starts, n_new)
        return new_starts

    def sample_n(self, array, n):
        """ Sample n elements from "array". """
        num_elements = array.shape[0]
        inds = np.random.choice(num_elements, size=n, replace=False)
        return array[inds, ...]

    def reset(self, state=None, evaluate=False):
        ''' Wrapper function for handling which start state to use for the reset. '''
        # check which sampling_method to use for the reset
        if state is not None:
            result = super().reset(state, goal=self.goal)
        elif evaluate or self.sampling_method == 'uniform':
            result = super().reset(self.sample_uniform(), goal=self.goal)
        else:
            result = super().reset(self.sample_curriculum(), goal=self.goal)
        if self.do_rendering:
            self.render()
        self.episodes_steps.append(0)

        return result

    def sample_uniform(self):
        ''' Sample start state uniformly from feasible states. '''
        # sample a start state from a uniform start distribution
        sample_cnt = 0
        while True:
            sample_cnt += 1
            s = [np.random.uniform(low=-5, high=5),np.random.uniform(low=-5, high=5)]
            if self.is_feasible(s) and not self.is_in_goal_area(s):
                return s

    def sample_curriculum(self):
        ''' Sample start state using uniform distribution on the states in the curriculum.'''
        start_ind = np.random.choice(self.curriculum_starts.shape[0])
        self.current_start = start_ind
        self.start_counts[start_ind] += 1

        return self.curriculum_starts[start_ind]

    def is_in_goal_area(self, state):
        ''' Checks if "state" is in goal area. '''
        dist_to_goal_center = np.sqrt((state[0] - self.wrapped_env.current_goal[0])**2 +
                                      (state[1] - self.wrapped_env.current_goal[1])**2)
        return dist_to_goal_center < self.goal_radius

    def set_model(self, model):
        ''' Set the model which is required for the evaluation. Must be called in the
            update loop in PPO2 in OpeanAI's baselines '''
        self.model = model

    def evaluate(self):
        ''' Evaluates the current model. Uses the model set in set_model. '''

        if self.eval_runs <= 0:
            return self.get_current_obs()

        print("\n\nEvaluation started ... ")
        # remove current start from self.start_counts since the env is reset for evaluation
        if self.sampling_method != 'uniform':
            self.start_counts[self.current_start] -= 1
        current_eval_index = len(self.eval_results)
        current_eval_starts = []
        current_eval_results = []
        for i in range(self.eval_runs):
            print("\tFinished {} of {}".format(i, self.eval_runs), end='\r')
            obs = self.reset(evaluate=True)
            current_eval_starts.append((obs[0], obs[1]))
            done = False
            for s in range(self.max_env_timestep):
                actions, _, _, _ = self.model.step(obs)
                obs, _, done, _ = super().step(actions)
                if self.do_rendering:
                    self.render()
                if done:
                    break
            if done:
                current_eval_results.append(1)
            else:
                current_eval_results.append(0)
        self.eval_starts[current_eval_index] = current_eval_starts
        self.eval_results[current_eval_index] = current_eval_results
        print(" ... Evaluation finished. Avg of current evaluation: {0}\n".format(np.average(current_eval_results)))
        self.save()
        obs = self.reset(evaluate=False)
        return obs

    def save(self):
        ''' Save the evaluation results and the model. '''
        
        print("Saving model to:\n\t'{}'".format(self.model_file_path))
        self.model.save(self.model_file_path)
        
        print("Saving evaluations to: \n\t'{}'\n\t'{}'\n".format(self.eval_starts_file_name, self.eval_results_file_name))

        fh = open(self.eval_starts_file_name, "w")
        json.dump(self.eval_starts, fh)
        fh.close()

        fh = open(self.eval_results_file_name, "w")
        json.dump(self.eval_results, fh)
        fh.close()
        

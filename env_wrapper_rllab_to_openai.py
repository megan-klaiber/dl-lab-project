import json
import numpy as np
import os
import random

from curriculum.envs.maze.point_maze_env import PointMazeEnv
from gym.spaces import Box
import time

#from curriculum.envs.maze.point_env import PointEnv
# from baselines.common.vec_env import VecEnv

# class WrappedPointEnv(PointEnv, VecEnv):
#class WrappedPointEnv(PointEnv):
class WrappedPointMazeEnv(PointMazeEnv):

    def __init__(self):
        # from curriculum.envs.base import FixedStateGenerator
        # fixed_goal_generator = FixedStateGenerator(state=(4, 4))
        # super().__init__(coef_inner_rew=1.0, maze_id=11, goal_generator=fixed_goal_generator)
        super().__init__(coef_inner_rew=1.0, maze_id=11)
        self.num_envs = 1
        self.max_env_timestep = 500
        self.episodes_steps = []
        self.episodes_goal_reached = []
        # integer that saves the index of the current start during training
        self.current_start = None
        self.global_train_steps = 0
        self.sampling_method = 'uniform'
        self.eval_results = {}
        # intialize start state lists
        # first initialize curriculum_starts with states close to the goal state

        self.steps_per_curriculum = 50000
        self.do_rendering = False
        self.R_max = 0.9
        self.R_min = 0.1

        self.verbose = False
        self.goal = (4, 4)
        result = super().reset(goal=self.goal)

    def post_init_stuff(self, max_env_timestep, eval_runs=10,
                        sampling_method='uniform', do_rendering=False, steps_per_curriculum = 50000,
                        verbose=False):
        self.max_env_timestep = max_env_timestep
        self.eval_runs = eval_runs
        self.sampling_method = sampling_method # can be either 'good_starts', 'all_previous' or 'uniform'
        self.do_rendering = do_rendering
        self.steps_per_curriculum = steps_per_curriculum

        if self.sampling_method != 'uniform':
            self.curriculum_starts = self.sample_nearby([self.wrapped_env.current_goal])
            self.start_counts = np.zeros(self.curriculum_starts.shape[0])
            self.goal_counts = np.zeros(self.curriculum_starts.shape[0])

        self.verbose = verbose


    @property
    def observation_space(self):
        ob_space = super().observation_space
        return Box(low=ob_space.low, high=ob_space.high, dtype=np.float32)

    @property
    def action_space(self):
        ac_space = super().action_space
        return Box(low=ac_space.low, high=ac_space.high, dtype=np.float32)

    def step(self, actions, train=True):
        obs, rewards, dones, infos = super().step(actions)
        if self.do_rendering:
            self.render()
            # timestep = 0.01
            # speedup = 5
            # time.sleep(timestep / speedup)
        if train:
            self.global_train_steps += 1
            if (self.global_train_steps % self.steps_per_curriculum == 0):
                print("self.global_train_steps / self.steps_per_curriculum: {0}".format(
                        self.global_train_steps / self.steps_per_curriculum)
                    )
            if (self.global_train_steps % self.steps_per_curriculum == 0) \
               and self.sampling_method != 'uniform':
                # find good starts
                goal_reach_frequencies = self.goal_counts / self.start_counts
                if self.sampling_method == 'all_previous':
                    starts = self.curriculum_starts
                else:
                    starts = self.good_starts(self.curriculum_starts, goal_reach_frequencies)
                self.curriculum_starts = self.sample_nearby(starts)

                self.start_counts = np.zeros(self.curriculum_starts.shape[0])
                self.goal_counts = np.zeros(self.curriculum_starts.shape[0])
            
        self.episodes_steps[-1] += 1
        # print(dones[0])
        if dones or (self.episodes_steps[-1] >= self.max_env_timestep):
            if dones:
                self.episodes_goal_reached.append(True)
                if train and self.sampling_method != 'uniform':
                    self.goal_counts[self.current_start] += 1
            else:
                self.episodes_goal_reached.append(False)
            # Reset env when goal is reached or max timesteps is reached.
            # print("Resetted in step!")
            if train:
                obs = self.reset(train=train)
        rewards = np.array([rewards])    # Maybe helps with command 'ev = explained_variance(values, returns)' in ppo2.py?
        dones = np.array([dones])    # Caused problems in runner.py at the end, when applying "sf01(mb_dones)".
        infos = {}   # Caused problems in runner.py, with part "for info in infos: ..."
        return obs, rewards, dones, infos

    def good_starts(self, states, success_freq):
        starts_good = np.array([states[i]  for i in range(len(states))
                        if (success_freq[i] > 0.1 and success_freq[i] < 0.9)])
        if self.verbose:
            print('good starts:', starts_good)
        return starts_good


    def sample_nearby(self, states, n_new=200, variance=0.5, t_b=50, M=1000):
        """
            n_new: number of sampled starts in the ned
            t_b:   horizon of one trajectory
            M:     number of state list after adding starts
        """
        # possibly add: radius which tests if sampled states are in a given
        # radius around the sampled start state or around the goal?

        starts = np.array(states)
        # if len(starts) == 1:
        #     print('Only one state in starts')
        #     ultimate_goal = starts[0]
        while(len(starts) < M):
            # number of starts given by first dimension of starts
            # num_starts = starts.shape[0]
            # s_0 = starts[np.random.choice(num_starts),...]
            s_0 = random.choice(starts)
            # print('s_0:',s_0)
            # print('Start state equals ultimate goal:', s_0==ultimate_goal)
            super().reset(init_state=s_0)
            for i in range(t_b):
                a = np.random.normal(scale=variance, size=self.action_dim)
                # only want the current observation
                _obs, _rew, _done, _env_info = super().step(a)
                # only want the start state part of the current state
                s_1 = self.get_current_obs()[:2].reshape(1, -1)
                starts = np.append(starts, s_1, axis=0)
                self.render()
                timestep = 0.01
                speedup = 1
                # time.sleep(timestep / speedup)
                # rllab also tested if the exploration led through the goal state
                # do not see why this is necessary atm

        # now sample from the M start states
        new_starts = self.sample_n(starts, n_new)
        return new_starts

    def sample_n(self, array, n):
        """ sample n elements from the array """
        num_elements = array.shape[0]
        inds = np.random.choice(num_elements, size=n, replace=False)
        return array[inds, ...]

    def reset(self, state=None, train=True):
        if state is not None:
            result = super().reset(state, goal=self.goal)
        elif not train or self.sampling_method == 'uniform':
            result = super().reset(self.sample_uniform(), goal=self.goal)
        elif train:
            result = super().reset(self.sample_curriculum(), goal=self.goal)
        else:
            result = super().reset(goal=self.goal)

        self.episodes_steps.append(0)

        return result

    def sample_uniform(self):
        # sample a start state from a uniform start distribution
        sample_cnt = 0
        while True:
            sample_cnt += 1
            s = [np.random.uniform(low=-5, high=5),np.random.uniform(low=-5, high=5)]
            if self.is_feasible(s):
                if self.verbose:
                    print('Samples until feasible:', sample_cnt)
                return s

    def sample_curriculum(self):
        # sample from the current start state distribution according to the curriculum
        start_ind = np.random.choice(self.curriculum_starts.shape[0])
        self.current_start = start_ind
        self.start_counts[start_ind] += 1
        return self.curriculum_starts[start_ind]

    def evaluate(self, model):
        # For using this method add add "runner.obs[:] = env.evaluate(model)" in ppo2.py.

        print("\n\nEvaluation started ... ")
        current_eval_index = len(self.eval_results)
        current_eval_results = []
        for i in range(self.eval_runs):
            obs = self.reset(train=False)
            done = False
            for s in range(self.max_env_timestep):
                actions, _, _, _ = model.step(obs)
                obs, _, done, _ = self.step(actions, train=False)
                if done:
                    break
            if done:
                current_eval_results.append(1)
            else:
                current_eval_results.append(0)
        self.eval_results[current_eval_index] = current_eval_results
        print(" ... Evaluation finished\n\n")
        obs = self.reset(train=True)
        return obs

    def save(self, file_name="results.json"):
        if not os.path.exists("results"):
            os.mkdir("results")

        fname = os.path.join("results", file_name)

        fh = open(fname, "w")
        json.dump(self.eval_results, fh)
        fh.close()

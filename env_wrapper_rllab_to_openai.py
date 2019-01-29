import numpy as np
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
        self.sampling_method = 'good_starts'
        self.eval_results = []
        # intialize start state lists
        # first initialize curriculum_starts with states close to the goal state
        self.curriculum_starts = self.sample_nearby([self.wrapped_env.current_goal])
        self.do_rendering = False
        
        self.start_counts = np.zeros(self.curriculum_starts.shape[0])
        self.goal_counts = np.zeros(self.curriculum_starts.shape[0])


    def post_init_stuff(self, max_env_timestep, eval_runs=10,
                        sampling_method='good_starts', do_rendering=False):
        self.max_env_timestep = max_env_timestep
        self.eval_runs = eval_runs
        self.sampling_method = sampling_method # can be either 'good_starts', 'all_previous' or 'uniform'
        self.do_rendering = do_rendering


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
            timestep = 0.01
            speedup = 5
            time.sleep(timestep / speedup)
        if train:
            if self.global_train_steps % 50000 == 0:
                # find good starts
                goal_reach_frequencies = self.goal_counts / self.start_counts
                # good_starts = ....
                starts = self.curriculum_starts
                self.curriculum_starts = self.sample_nearby(starts) # or good_starts...
                
                self.start_counts = np.zeros(self.curriculum_starts.shape[0])
                self.goal_counts = np.zeros(self.curriculum_starts.shape[0])

            else:
                self.global_train_steps += 1
        infos = {}   # Caused problems in runner.py, with part "for info in infos: ..."
        dones = np.array([dones])    # Caused problems in runner.py at the end, when applying "sf01(mb_dones)".
        rewards = np.array([rewards])    # Maybe helps with command 'ev = explained_variance(values, returns)' in ppo2.py?
        self.episodes_steps[-1] += 1
        # print(dones[0])
        if dones[0] or (self.episodes_steps[-1] >= self.max_env_timestep):
            if dones[0]:
                self.episodes_goal_reached.append(True)
                if train:
                    self.goal_counts[self.current_start] += 1
            else:
                self.episodes_goal_reached.append(False)
            # Reset env when goal is reached or max timesteps is reached.
            # print("Resetted in step!")
            obs = self.reset(train)
        return obs, rewards, dones, infos

    def sample_nearby(self, states, n_new=200, variance=0.5, t_b=50, M=1000):
        """
            n_new: number of sampled starts in the ned
            t_b:   horizon of one trajectory
            M:     number of state list after adding starts
        """
        # possibly add: radius which tests if sampled states are in a given
        # radius around the sampled start state or around the goal?

        starts = np.array(states)
        if len(starts) == 1:
            print('Only one state in starts')
            ultimate_goal = starts[0]
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
                time.sleep(timestep / speedup)
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
            result = super().reset(state)
        elif train:
            result = super().reset(self.sample_curriculum())
        elif not train:
            result = super().reset(self.sample_uniform())
        # elif self.fixed_restart_state is not None:
        #     result = super().reset(self.fixed_restart_state)
        else:
            result = super().reset()

        self.episodes_steps.append(0)
        return result

    def sample_uniform(self):
        # sample a start state from a uniform start distribution
        sample_cnt = 0
        while True:
            sample_cnt += 1
            s = [np.random.uniform(low=-5, high=5),np.random.uniform(low=-5, high=5)]
            if self.is_feasible(s):
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

        current_eval_results = []
        for i in range(self.eval_runs):
            obs = self.reset(train=False)
            done = False
            for s in range(self.max_env_timestep):
                actions, _, _, _ = model.step(obs)
                obs, _, done, _ = self.step(actions)
                if done:
                    break
            current_eval_results.append(done)
        self.eval_results.add(current_eval_results)
        obs = self.reset(train=True)
        return obs

    def save(self, file_name="results.json"):
        if not os.path.exists("results"):
            os.mkdir("results")

        fname = os.path.join("results", file_name)
        fh = open(fname, "w")
        json.dump(self.eval_results, fh)
        fh.close()

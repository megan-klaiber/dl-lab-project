import numpy as np

from curriculum.envs.maze.point_maze_env import PointMazeEnv
from gym.spaces import Box

#from curriculum.envs.maze.point_env import PointEnv
# from baselines.common.vec_env import VecEnv

# class WrappedPointEnv(PointEnv, VecEnv):
#class WrappedPointEnv(PointEnv):
class WrappedPointMazeEnv(PointMazeEnv):
    
    def __init__(self):
        super().__init__(coef_inner_rew=1.0, maze_id=11)
        self.num_envs = 1
        self.max_env_timestep = 500
        self.fixed_restart_state = None
        self.episodes_steps = []
        self.episodes_goal_reached = []

    def post_init_stuff(self, max_env_timestep, fixed_restart_state, eval_runs=10):
        self.max_env_timestep = max_env_timestep
        self.fixed_restart_state = fixed_restart_state
        self.eval_runs = eval_runs

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
        infos = {}   # Caused problems in runner.py, with part "for info in infos: ..."
        dones = np.array([dones])    # Caused problems in runner.py at the end, when applying "sf01(mb_dones)".
        rewards = np.array([rewards])    # Maybe helps with command 'ev = explained_variance(values, returns)' in ppo2.py?
        self.episodes_steps[-1] += 1
        # print(dones[0])
        if dones[0] or (self.episodes_steps[-1] >= self.max_env_timestep):
            if dones[0]:
                self.episodes_goal_reached.append(True)
            else:
                self.episodes_goal_reached.append(False)
            # Reset env when goal is reached or max timesteps is reached.
            # print("Resetted in step!")
            obs = self.reset(train)
        return obs, rewards, dones, infos

    def reset(self, state=None, train=True):
        if state is not None:
            result = super().reset(state)
        elif self.fixed_restart_state is not None:
            result = super().reset(self.fixed_restart_state)
        else:
            result = super().reset()
            
        self.episodes_steps.append(0)
        return result
    
    def evaluate(self, model):
        for i in range(self.eval_runs):
            self.reset(train=False)
            for s in range(self.max_env_timestep):
                model
        # env_eval = WrappedPointMazeEnv()
        # env_eval.setstartdistr()
        # self.reset()
        pass

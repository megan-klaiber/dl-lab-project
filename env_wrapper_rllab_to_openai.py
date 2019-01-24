import numpy as np

from curriculum.envs.maze.point_maze_env import PointMazeEnv
from gym.spaces import Box

#from curriculum.envs.maze.point_env import PointEnv
# from baselines.common.vec_env import VecEnv

# class WrappedPointEnv(PointEnv, VecEnv):
#class WrappedPointEnv(PointEnv):
class WrappedPointMazeEnv(PointMazeEnv):
    
    def __init__(self):
        super().__init__()
        self.num_envs = 1

    @property
    def observation_space(self):
        ob_space = super().observation_space
        return Box(low=ob_space.low, high=ob_space.high, dtype=np.float32)

    @property
    def action_space(self):
        ac_space = super().action_space
        return Box(low=ac_space.low, high=ac_space.high, dtype=np.float32)

    

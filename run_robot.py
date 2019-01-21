import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=0, type=int, nargs=1, help="Defines the used environment: 0='Ring on Peg task'; 1='Key insertion task'")
    parser.add_argument("--max_time_steps", default=1000, type=int, nargs=1, help="Maximal number of timesteps (default=1000).")
    args = parser.parse_args()

    if type(args.env) == list:
        args.env = args.env[0]
    if type(args.max_time_steps) == list:
        args.max_time_steps = args.max_time_steps[0]

    if args.env == 0:
        from curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv
        env = Arm3dDiscEnv()
    elif args.env == 1:
        from curriculum.envs.arm3d.arm3d_key_env import Arm3dKeyEnv
        env = Arm3dKeyEnv()
    elif args.env == 2:
        # For Testing
        from curriculum.envs.maze.point_env import PointEnv
        env = PointEnv()
    elif args.env == 3:
        # For Testing
        from curriculum.envs.maze.point_maze_env import PointMazeEnv
        env = PointMazeEnv()
    else:
        raise ValueError("Unknown value for parameter 'env'.")

    env.render()

    do_action = True
    robot = (args.env == 0) or (args.env == 1)
    
    for i in range(args.max_time_steps):
        print("Step {} ...".format(i))
        if do_action and (i % 10 == 0):
            if robot:
                a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            else:
                a = np.array([0.5, 0.5])
            env.step(action=a)
        env.render()


    # Alogrithm 1 from [1]



    # from baselines.trpo_mpi import trpo_mpi
    # trpo_mpi.learn(network='mlp', env=env, total_timesteps=2)

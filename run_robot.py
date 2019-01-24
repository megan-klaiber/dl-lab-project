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

    #while(True):
        #k = raw_input('> ')
        #if k == 'q':
            #break;


    env.reset()
    env.render()

    do_action = True
    robot = (args.env == 0) or (args.env == 1)

    # while(True):
    # observation, reward, done, info = env.step(action=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
    # # action: turn robot (positive = left), lower arm (positive = lower), turn arm (positive = backward), elbow joint (positive = straighten), forearm (positive = outwards), wrist, turn hand
    # observation, reward, done, info = env.step(action=np.array([0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]))
    # observation, reward, done, info = env.step(action=np.array([0.00, 0.00, -0.01, 0.00, 0.00, 0.00, 0.0]))
    # print("reward: {}".format(reward))
    # env.render()

    for i in range(5000):
        # action: turn robot (positive = left), lower arm (positive = lower), turn arm (positive = backward), elbow joint (positive = straighten), forearm (positive = outwards), wrist, turn hand
        print("step: {}".format(i))
        observation, reward, done, info = env.step(action=np.array([+1, 0.00, -0.0, -0.00, 0.00, 0.00, 0.0]))
        print("observation: {}".format(observation))
        print("reward: {}".format(reward))

    # for i in range(args.max_time_steps):
    #     print("Step {} ...".format(i))
    #     if do_action and (i % 10 == 0):
    #         if robot:
    #             a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    #         else:
    #             a = np.array([0.5, 0.5])
    #         env.step(action=a)
        # env.render()

    # for i in range(5000):
    #     # action: turn robot (positive = left), lower arm (positive = lower), turn arm (positive = backward), elbow joint (positive = straighten), forearm (positive = outwards), wrist, turn hand
    #     print("step: {}".format(i))
    #     observation, reward, done, info = env.step(action=np.array([0, 1, -0.0, -0.00, 0.00, 0.00, 0.0]))
    #     print("observation: {}".format(observation))
    #     print("reward: {}".format(reward))
    #     env.render()
    #
    # for i in range(5000):
    #     # action: turn robot (positive = left), lower arm (positive = lower), turn arm (positive = backward), elbow joint (positive = straighten), forearm (positive = outwards), wrist, turn hand
    #     print("step: {}".format(i))
    #     observation, reward, done, info = env.step(action=np.array([0, 0.00, -0.0, -0.00, 1, 0.00, 0.0]))
    #     print("observation: {}".format(observation))
    #     print("reward: {}".format(reward))
    #     env.render()

    #while(True):
    #arm3d_env.step(action=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
    #arm3d_env.render()

    # arm3d_env.step(action=a)

    # inner_env = normalize(arm3d_env)
    i = 15000
    while True:
        i = i + 1
        print("step: {}".format(i))
        observation, reward, done, info = env.step(action=np.array([0.00, 0.00, -0.0, -0.00, 0.00, 0.00, 0.0]))
        print("observation: {}".format(observation))
        env.render()

    # Alogrithm 1 from [1]

    if not robot:    # Maze
        # Get goal state.
        current_goal = env.current_goal

        n = 10
        length = np.random.uniform(0, 0.1, n)
        angle = np.pi * np.random.uniform(0, 2, n)
        x = np.sqrt(length) * np.cos(angle)
        y = np.sqrt(length) * np.sin(angle)
        # import matplotlib.pyplot as plt
        # plt.scatter(x, y); plt.show()

        starts_old = np.stack((x, y), axis=1)
        starts = starts_old
        rews = np.ones(n)

    iter_num = 1000
    N_new = 20
    for i to iter_num:
        starts = sample_nearby(starts, N_new)

    # from baselines.trpo_mpi import trpo_mpi
    # trpo_mpi.learn(network='mlp', env=env, total_timesteps=2)

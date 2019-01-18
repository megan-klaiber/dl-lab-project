import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=0, type=int, nargs=1, help=("Defines the used environment: 0='Ring on Peg task'; 1='Key insertion task'"))
    args = parser.parse_args()

    if type(args.env) == list:
        args.env = args.env[0]

    if args.env == 0:
        from curriculum.envs.arm3d.arm3d_move_peg_env import Arm3dMovePegEnv
        env = Arm3dMovePegEnv()
    elif args.env == 1:
        from curriculum.envs.arm3d.arm3d_key_env import Arm3dKeyEnv
        env = Arm3dKeyEnv()
    else:
        raise ValueError("Unknown value for parameter 'env'.")

    #while(True):
        #k = raw_input('> ')
        #if k == 'q':
            #break;
        
    # env.step(action=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
    while(True):
        env.render()
    



# from rllab.envs.normalized_env import normalize


# arm3d_env = Arm3dKeyEnv(ctrl_cost_coeff=[0])
# arm3d_env = Arm3dKeyEnv(ctrl_cost_coeff=0)

#while(True):
    #arm3d_env.step(action=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
    #arm3d_env.render()
    
# arm3d_env.step(action=a)

# inner_env = normalize(arm3d_env)

# arm3d_env.step(action=np.array(0.1, 0, 0, 0, 0, 0, 0))

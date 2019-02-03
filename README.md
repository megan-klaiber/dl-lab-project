# dl-lab-project
Final project for the Deep Learning lab course of the University of Freiburg.

# Installation notes

Download code from ​ https://github.com/florensacc/rllab-curriculum and then follow the instructions to compile the code as described in https://rllab.readthedocs.io/en/latest/.

Install OpenAI's baseline algorithms as described in https://github.com/openai/baselines. Make sure to install it in the conda environment created in the rllab installation. Also make sure that python uses the OpenAI baselines when importing baselines and not the baselines from rllab.

In `baselines/baselines/ppo2/ppo2.py` add the line `env.set_model(model)` after line 125 (`for update in range(1, nupdates+1):`).

To test the installation execute `python ../dl-lab-project/run_ppo.py --outer_iter=2` from `rllab-curriculum-master/`. This assumes that you have the dl-lab-project repository and the rllab-curriculum repository in the same folder. Otherwise adapt the path to get to the run_ppo.py file from within the rllab-curriculum-master folder. 









#!/bin/bash

echo 'uniform sampling'
python ../dl-lab-project/run_ppo.py --sampling_method='uniform' --seed=10
python ../dl-lab-project/run_ppo.py --sampling_method='uniform' --seed=20
python ../dl-lab-project/run_ppo.py --sampling_method='uniform' --seed=30
python ../dl-lab-project/run_ppo.py --sampling_method='uniform' --seed=40
python ../dl-lab-project/run_ppo.py --sampling_method='uniform' --seed=50

echo 'good starts'
python ../dl-lab-project/run_ppo.py --sampling_method='good_starts' --seed=10
python ../dl-lab-project/run_ppo.py --sampling_method='good_starts' --seed=20
python ../dl-lab-project/run_ppo.py --sampling_method='good_starts' --seed=30
python ../dl-lab-project/run_ppo.py --sampling_method='good_starts' --seed=40
python ../dl-lab-project/run_ppo.py --sampling_method='good_starts' --seed=50

echo 'all previous'
python ../dl-lab-project/run_ppo.py --sampling_method='all_previous' --seed=10
python ../dl-lab-project/run_ppo.py --sampling_method='all_previous' --seed=20
python ../dl-lab-project/run_ppo.py --sampling_method='all_previous' --seed=30
python ../dl-lab-project/run_ppo.py --sampling_method='all_previous' --seed=40
python ../dl-lab-project/run_ppo.py --sampling_method='all_previous' --seed=50

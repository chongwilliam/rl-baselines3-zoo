#!/bin/bash
# Training script launch point 

python train.py --algo ppo --env force_env-v0 -P --tensorboard-log /tmp/stable-baselines/ --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 --save-freq 100000
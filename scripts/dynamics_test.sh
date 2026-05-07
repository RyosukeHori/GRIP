#!/bin/bash

python dynamics_net/run_hydra.py \
    exp_name=DynamicsNet\
    env=env_im_pnn_grip \
    learning=im_pnn_big \
    robot=smpl_humanoid \
    robot.freeze_hand=True \
    robot.box_body=False \
    robot.real_weight_porpotion_boxes=False \
    env.motion_file=data/preprocessed/dynamics_net/test \
    env.num_imus=4 \
    env.load.object=True \
    env.training_prim=0 \
    env.num_envs=508 \
    env.max_len=500 \
    env.episode_length=500 \
    env.failed_keys="" \
    env.rand_seq=False \
    env.fall_recovery=True \
    env.enableEarlyTermination=False \
    env.VerticalDistanceTermination=True \
    env.terminationDistance=0.5 \
    env.reset_bodies=['Pelvis'] \
    env.trackBodies=[] \
    env.win_len=1 \
    rl_device="cuda:0" \
    device_id=0 \
    learning.params.config.device="cuda:0" \
    epoch=-1 \
    test=True \
    has_eval=True \
    im_eval=True \
    no_log=True \
    headless=True


#!/bin/bash
# Two-stage DynamicsNet training.
#
# Stage 1 (motion imitation):
#   - From-scratch (epoch=0)
#   - 1000-frame episodes with early termination on a vertical fall
#     (enableEarlyTermination=True, VerticalDistanceTermination=True,
#     terminationDistance=0.5) so the agent first learns to track the
#     reference motion without falling.
#   - Stops automatically at max_epochs=10000.
#
# Stage 2 (fall recovery):
#   - Resumes from the latest stage-1 checkpoint (epoch=-1)
#   - 500-frame episodes with all early termination disabled, so the agent
#     keeps stepping even after a fall and learns to recover.
#   - Runs until you Ctrl-C (max_epochs left at the config default).

set -e

python dynamics_net/run_hydra.py \
    exp_name=DynamicsNet \
    env=env_im_pnn_grip \
    learning=im_pnn_big \
    robot=smpl_humanoid \
    robot.freeze_hand=True \
    robot.box_body=False \
    robot.real_weight_porpotion_boxes=False \
    env.motion_file=data/preprocessed/dynamics_net/train \
    env.num_imus=4 \
    env.load.object=True \
    env.training_prim=0 \
    env.num_envs=2048 \
    env.max_len=1000 \
    env.episode_length=1000 \
    env.subjects=[] \
    env.failed_keys="" \
    env.enableEarlyTermination=True \
    env.VerticalDistanceTermination=True \
    env.terminationDistance=0.5 \
    env.trackBodies=[] \
    env.win_len=1 \
    env.fall_penalty=0.0 \
    rl_device="cuda:0" \
    device_id=0 \
    learning.params.config.device="cuda:0" \
    learning.params.config.max_epochs=10000 \
    learning.params.config.minibatch_size=32768 \
    learning.params.config.amp_batch_size=1024 \
    learning.params.config.amp_minibatch_size=8192 \
    no_log=False \
    headless=True \
    im_eval=True \
    has_eval=True \
    epoch=0


python dynamics_net/run_hydra.py \
    exp_name=DynamicsNet \
    env=env_im_pnn_grip \
    learning=im_pnn_big \
    robot=smpl_humanoid \
    robot.freeze_hand=True \
    robot.box_body=False \
    robot.real_weight_porpotion_boxes=False \
    env.motion_file=data/preprocessed/dynamics_net/train \
    env.num_imus=4 \
    env.load.object=True \
    env.training_prim=0 \
    env.num_envs=4096 \
    env.max_len=500 \
    env.episode_length=500 \
    env.subjects=[] \
    env.failed_keys="" \
    env.enableEarlyTermination=False \
    env.VerticalDistanceTermination=False \
    env.terminationDistance=0.25 \
    env.trackBodies=[] \
    env.win_len=1 \
    env.fall_penalty=0.0 \
    rl_device="cuda:0" \
    device_id=0 \
    learning.params.config.device="cuda:0" \
    learning.params.config.minibatch_size=32768 \
    learning.params.config.amp_batch_size=1024 \
    learning.params.config.amp_minibatch_size=8192 \
    no_log=False \
    headless=True \
    im_eval=True \
    has_eval=True \
    epoch=-1

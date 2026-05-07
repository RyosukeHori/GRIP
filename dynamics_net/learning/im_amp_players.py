import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
import shutil

import numpy as np
import torch
from dynamics_net.utils.flags import flags
from rl_games.algos_torch import torch_ext
from dynamics_net.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import dynamics_net.learning.amp_players as amp_players
from tqdm import tqdm
import joblib
import time
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
from rl_games.common.tr_helpers import unsqueeze_obs

from scipy.spatial.transform import Rotation as sRot
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

COLLECT_Z = False

class IMAMPPlayerContinuous(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)

        self.terminate_state = torch.zeros(self.env.task.num_envs, device=self.device)
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.gt_rot, self.gt_rot_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.pred_rot, self.pred_rot_all = [], []
        self.torque, self.torque_all = [], []
        self.grf, self.grf_all = [], []
        self.disc_probs, self.disc_probs_all = [], []
        self.skeleton_trees_all = []
        self.curr_stpes = 0

        if COLLECT_Z:
            self.zs, self.zs_all = [], []

        humanoid_env = self.env.task
        # humanoid_env._termination_distances[:] = 0.5 # if not humanoid_env.strict_eval else 0.25 # ZL: use UHC's termination distance
        humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0
        
        # Set discriminator on the environment
        if hasattr(self.model.a2c_network, 'eval_disc'):
            disc_model = self.model.a2c_network.eval_disc
            amp_input_mean_std = self._amp_input_mean_std if self._normalize_amp_input else None
            humanoid_env.set_discriminator(disc_model, amp_input_mean_std)

        if flags.im_eval:
            self.success_rate = 0
            self.pbar = tqdm(range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs))
            humanoid_env.zero_out_far = False
            humanoid_env.zero_out_far_train = False
            
            if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
            
            humanoid_env.cycle_motion = False
            self.print_stats = False
        
        # joblib.dump({"mlp": self.model.a2c_network.actor_mlp, "mu": self.model.a2c_network.mu}, "single_model.pkl") # ZL: for saving part of the model.
        return

    def _post_step(self, info, done):
        super()._post_step(info)
        
        
        # modify done such that games will exit and reset.
        if flags.im_eval:

            humanoid_env = self.env.task
            
            termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            # termination_state = info["terminate"]
            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
            if (~self.terminate_state).sum() > 0:
                max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
                curr_ids = humanoid_env._motion_lib._curr_motion_ids
                if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
                    bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    if (~self.terminate_state[:bound]).sum() > 0:
                        curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                    else:
                        curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
                else:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"])
            self.pred_pos.append(info["body_pos"])
            self.gt_rot.append(info["body_rot_gt"])
            self.pred_rot.append(info["body_rot"])
            self.torque.append(info["torque"])
            self.grf.append(info["grf_sim"])
            
            # History correction triggered by fall detection
            if 'fall_correction' in info and info['fall_correction'] is not None:
                fall_info = info['fall_correction']
                fallen_env_ids = fall_info['env_ids']
                buf_pos_history = fall_info['buf_pos']  # (N, 100, 24, 3)
                buf_rot_history = fall_info['buf_rot']  # (N, 100, 24, 4)
                buf_grf_history = fall_info['buf_grf']  # (N, 100, 24, 3)
                current_step = fall_info['current_step']

                # Correct the last 100 frames
                for i, env_id in enumerate(fallen_env_ids):
                    # Compute the correction range (from current_step - 99 to current_step)
                    start_frame = max(0, current_step - 99)
                    end_frame = current_step + 1
                    frames_to_correct = end_frame - start_frame

                    if frames_to_correct > 0 and start_frame < len(self.pred_pos):
                        # Replace the matching frames in history with the buffered estimates
                        history_offset = 100 - frames_to_correct  # start index inside buf_history

                        for frame_idx in range(frames_to_correct):
                            abs_frame_idx = start_frame + frame_idx
                            if abs_frame_idx < len(self.pred_pos):
                                self.pred_pos[abs_frame_idx][env_id] = buf_pos_history[i, history_offset + frame_idx]
                                self.pred_rot[abs_frame_idx][env_id] = buf_rot_history[i, history_offset + frame_idx]
                                self.grf[abs_frame_idx][env_id] = buf_grf_history[i, history_offset + frame_idx]

                # print(f"  -> Corrected last 100 frames of history for {len(fallen_env_ids)} envs (replaced with estimates)")

            # Discriminator evaluation
            amp_obs = info['amp_obs'].float().to(self.device)
            with torch.no_grad():
                # Normalize (same as during training)
                if self._normalize_amp_input:
                    amp_obs = self._amp_input_mean_std(amp_obs)
                
                disc_logits = self.model.a2c_network.eval_disc(amp_obs)
                disc_probs = torch.sigmoid(disc_logits).squeeze()
                
                self.disc_probs.append(disc_probs.cpu().numpy())
            
            if COLLECT_Z: self.zs.append(info["z"])
            self.curr_stpes += 1

            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                
                self.terminate_memory.append(self.terminate_state.cpu().numpy())
                self.success_rate = (1 - np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())

                # MPJPE
                all_mpjpe = torch.stack(self.mpjpe)
                try:
                    assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
                except:
                    import ipdb; ipdb.set_trace()
                    print('??')

                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())] # -1 since we do not count the first frame. 
                
                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_body_rot_pred = np.stack(self.pred_rot)
                all_body_rot_pred = [all_body_rot_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_body_rot_gt = np.stack(self.gt_rot)
                all_body_rot_gt = [all_body_rot_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_torque = np.stack(self.torque)
                all_torque = [all_torque[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_grf = np.stack(self.grf)
                all_grf = [all_grf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                
                all_disc_probs = np.stack(self.disc_probs)
                if len(all_disc_probs.shape) == 1:
                    all_disc_probs = all_disc_probs.reshape(-1, 1)
                all_disc_probs = [all_disc_probs[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

                if COLLECT_Z:
                    all_zs = torch.stack(self.zs)
                    all_zs = [all_zs[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.zs_all += all_zs

                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt
                self.pred_rot_all += all_body_rot_pred
                self.gt_rot_all += all_body_rot_gt
                self.skeleton_trees_all += humanoid_env.skeleton_trees
                self.torque_all += all_torque
                self.grf_all += all_grf
                self.disc_probs_all += all_disc_probs

                if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                    terminate_hist = np.concatenate(self.terminate_memory)
                    succ_idxes = np.nonzero(~terminate_hist[: humanoid_env._motion_lib._num_unique_motions])[0].tolist()
                    failed_idxes = np.nonzero(terminate_hist[: humanoid_env._motion_lib._num_unique_motions])[0].tolist()

                    pred_pos_all_succ = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_pos_all_succ = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    pred_pos_all_failed = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in failed_idxes]
                    gt_pos_all_failed = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in failed_idxes]
                    pred_pos_all = self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]
                    gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]

                    pred_rot_all_succ = [(self.pred_rot_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_rot_all_succ = [(self.gt_rot_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    pred_rot_all_failed = [(self.pred_rot_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in failed_idxes]
                    gt_rot_all_failed = [(self.gt_rot_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in failed_idxes]
                    # pred_rot_all = self.pred_rot_all[:humanoid_env._motion_lib._num_unique_motions]
                    # gt_rot_all = self.gt_rot_all[: humanoid_env._motion_lib._num_unique_motions]
                    all_torque_succ = [(self.torque_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    all_torque_failed = [(self.torque_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in failed_idxes]

                    all_grf_succ = [(self.grf_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    all_grf_failed = [(self.grf_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in failed_idxes]
                    
                    # Categorize discriminator results
                    all_disc_probs_succ = [(self.disc_probs_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    all_disc_probs_failed = [(self.disc_probs_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in failed_idxes]
                    
                    # print("skeleton trees size: ", len(humanoid_env.skeleton_trees))
                    # print("terminate_hist: ", terminate_hist)
                    # print("terminate_hist shape: ", terminate_hist.shape)

                    skeleton_trees_failed = [self.skeleton_trees_all[i] for i in failed_idxes]
                    skeleton_trees_succ = [self.skeleton_trees_all[i] for i in succ_idxes]

                    # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                    # humanoid_env._motion_lib.get_motion_num_steps().sum()

                    failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
                    if flags.real_traj:
                        pred_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all]
                        gt_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all]
                        pred_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all_succ]
                        gt_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all_succ]
                        
                        
                        
                    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
                    metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                    metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    min_frames = humanoid_env._motion_lib.get_motion_num_steps().min()
                    max_frames = humanoid_env._motion_lib.get_motion_num_steps().max()
                    mean_frames = humanoid_env._motion_lib.get_motion_num_steps().float().mean()
                    std_frames = humanoid_env._motion_lib.get_motion_num_steps().float().std()

                    print("------------------------------------------")
                    print("------------------------------------------")
                    print(f"Success Rate: {self.success_rate:.10f}")
                    print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                    print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
                    print(f"Min frames: {min_frames} | Max frames: {max_frames} | Mean frames: {mean_frames:.3f} | Std frames: {std_frames:.3f}")
                    # print(1 - self.terminate_state.sum() / self.terminate_state.shape[0])

                    print(f"Total Seq Num: {humanoid_env._motion_lib._num_unique_motions} | Failed Num: {len(failed_idxes)} | Success Num: {len(succ_idxes)}")

                    print(self.config['network_path'])
                    if COLLECT_Z:
                        zs_all = self.zs_all[:humanoid_env._motion_lib._num_unique_motions]
                        zs_dump = {k: zs_all[idx].cpu().numpy() for idx, k in enumerate(humanoid_env._motion_lib._motion_data_keys)}
                        joblib.dump(zs_dump, osp.join(self.config['network_path'], "zs_run.pkl"))

                    # import ipdb; ipdb.set_trace()

                    # joblib.dump(np.concatenate(self.zs_all[: humanoid_env._motion_lib._num_unique_motions]), osp.join(self.config['network_path'], "zs.pkl"))

                    joblib.dump(failed_keys, osp.join(self.config['network_path'], "failed.pkl"))
                    joblib.dump(success_keys, osp.join(self.config['network_path'], "long_succ.pkl"))
                    
                    print("Saving results....")

                    # Process success rotations
                    if len(success_keys) > 0:
                        save_dir = osp.join(self.config['network_path'], "results_success")
                        if osp.exists(save_dir):
                            shutil.rmtree(save_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        for idx in range(len(success_keys)):
                            pred_pos = pred_pos_all_succ[idx]
                            gt_pos = gt_pos_all_succ[idx]
                            T, J, _ = pred_pos.shape
                            root_trans = torch.from_numpy(pred_pos[:, 0, :])
                            offset = skeleton_trees_succ[0].local_translation[0]
                            root_trans_offset = root_trans - offset
                            success_quat = pred_rot_all_succ[idx].reshape(-1, 4)
                            success_quat = (sRot.from_quat(success_quat.reshape(-1, 4)) * humanoid_env.pre_rot).as_quat().reshape(T, -1, 4)
                            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees_succ[0], torch.from_numpy(success_quat), root_trans_offset, is_local=False)
                            local_rot = new_sk_state.local_rotation
                            success_mat = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_matrix().reshape(T, J, 3, 3)
                            pred_rot = success_mat[:, humanoid_env.mujoco_2_smpl, :]

                            root_trans = torch.from_numpy(gt_pos[:, 0, :])
                            offset = skeleton_trees_succ[0].local_translation[0]
                            root_trans_offset = root_trans - offset
                            success_quat = gt_rot_all_succ[idx].reshape(-1, 4)
                            success_quat = (sRot.from_quat(success_quat.reshape(-1, 4)) * humanoid_env.pre_rot).as_quat().reshape(T, -1, 4)
                            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees_succ[0], torch.from_numpy(success_quat), root_trans_offset, is_local=False)
                            local_rot = new_sk_state.local_rotation
                            success_mat = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_matrix().reshape(T, J, 3, 3)
                            gt_rot = success_mat[:, humanoid_env.mujoco_2_smpl, :]

                            success_torque = np.zeros((T, 24, 3))
                            success_torque[:, 1:] = all_torque_succ[idx].reshape(-1, 23, 3)
                            success_torque = success_torque[:, humanoid_env.mujoco_2_smpl, :]

                            success_grf = all_grf_succ[idx]
                            success_grf = success_grf[:, humanoid_env.mujoco_2_smpl, :]
                            success_disc_probs = all_disc_probs_succ[idx]
                            
                            np.savez(osp.join(save_dir, f"{success_keys[idx]}.npz"), pred_pos=pred_pos, gt_pos=gt_pos, pred_rot=pred_rot, gt_rot=gt_rot, torque=success_torque, grf=success_grf, disc_probs=success_disc_probs)

                    # Process failed rotations
                    if len(failed_keys) > 0:
                        save_dir = osp.join(self.config['network_path'], "results_failed")
                        if osp.exists(save_dir):
                            shutil.rmtree(save_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        for idx in range(len(failed_keys)):
                            pred_pos = pred_pos_all_failed[idx]
                            gt_pos = gt_pos_all_failed[idx]
                            T, J, _ = pred_pos.shape
                            root_trans = torch.from_numpy(pred_pos[:, 0, :])
                            offset = skeleton_trees_failed[0].local_translation[0]
                            root_trans_offset = root_trans - offset
                            failed_quat = pred_rot_all_failed[idx].reshape(-1, 4)
                            failed_quat = (sRot.from_quat(failed_quat.reshape(-1, 4)) * humanoid_env.pre_rot).as_quat().reshape(T, -1, 4)
                            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees_failed[0], torch.from_numpy(failed_quat), root_trans_offset, is_local=False)
                            local_rot = new_sk_state.local_rotation
                            failed_mat = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_matrix().reshape(T, J, 3, 3)
                            pred_rot = failed_mat[:, humanoid_env.mujoco_2_smpl, :]

                            root_trans = torch.from_numpy(gt_pos[:, 0, :])
                            offset = skeleton_trees_failed[0].local_translation[0]
                            root_trans_offset = root_trans - offset
                            failed_quat = gt_rot_all_failed[idx].reshape(-1, 4)
                            failed_quat = (sRot.from_quat(failed_quat.reshape(-1, 4)) * humanoid_env.pre_rot).as_quat().reshape(T, -1, 4)
                            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees_failed[0], torch.from_numpy(failed_quat), root_trans_offset, is_local=False)
                            local_rot = new_sk_state.local_rotation
                            failed_mat = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_matrix().reshape(T, J, 3, 3)
                            gt_rot = failed_mat[:, humanoid_env.mujoco_2_smpl, :]

                            failed_torque = np.zeros((T, 24, 3))
                            failed_torque[:, 1:] = all_torque_failed[idx].reshape(-1, 23, 3)
                            failed_torque = failed_torque[:, humanoid_env.mujoco_2_smpl, :]

                            failed_grf = all_grf_failed[idx]
                            failed_grf = failed_grf[:, humanoid_env.mujoco_2_smpl, :]
                            failed_disc_probs = all_disc_probs_failed[idx]
                            
                            np.savez(osp.join(save_dir, f"{failed_keys[idx]}.npz"), pred_pos=pred_pos, gt_pos=gt_pos, pred_rot=pred_rot, gt_rot=gt_rot, torque=failed_torque, grf=failed_grf, disc_probs=failed_disc_probs)
                        

                done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

                humanoid_env.forward_motion_samples()
                self.terminate_state = torch.zeros(
                    self.env.task.num_envs, device=self.device
                )

                self.pbar.update(1)
                self.pbar.refresh()
                self.mpjpe, self.gt_pos, self.pred_pos, self.gt_rot, self.pred_rot = [], [], [], [], []
                self.torque, self.grf = [], []
                self.disc_probs = []
                if COLLECT_Z: self.zs = []
                self.curr_stpes = 0


            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)

        return done
    
    def get_z(self, obs_dict):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            z = self.model.a2c_network.eval_z(input_dict)
            return z

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break
            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)


                    if COLLECT_Z: z = self.get_z(obs_dict)
                        

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    if COLLECT_Z: info['z'] = z
                    done = self._post_step(info, done.clone())

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)
                        
                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = (
                                    s[:, all_done_indices, :] * 0.0
                                )

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if "battle_won" in info:
                                print_game_res = True
                                game_res = info.get("battle_won", 0.5)
                            if "scores" in info:
                                print_game_res = True
                                game_res = info.get("scores", 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count, "w:", game_res,)
                            else:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count,)

                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )

        return

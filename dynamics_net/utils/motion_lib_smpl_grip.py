import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from ast import If
import numpy as np
import yaml
from tqdm import tqdm

from isaacgym.torch_utils import *
from dynamics_net.utils import torch_utils
import joblib
import torch
from kinematics_net import articulate as art
from poselib.poselib.core.rotation3d import rot_matrix_from_quaternion, quat_from_rotation_matrix
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import torch.multiprocessing as mp
import copy
import gc
from dynamics_net.utils.flags import flags
import random
from scipy.spatial.transform import Rotation as sRot
from dynamics_net.utils.motion_lib_smpl import MotionLibSMPL
from dynamics_net.utils.motion_lib_base import compute_motion_dof_vels, DeviceCache, FixHeightMode
import cv2
from enum import Enum
from smpl_sim.utils.torch_ext import to_torch
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

# if not USE_CACHE:
#     old_numpy = torch.Tensor.numpy

#     class Patch:

#         def numpy(self):
#             if self.is_cuda:
#                 return self.to("cpu").numpy()
#             else:
#                 return old_numpy(self)

#     torch.Tensor.numpy = Patch.numpy


def local_rotation_to_dof_vel(local_rot0, local_rot1, dt):
    # Assume each joint is 3dof
    diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rot0), local_rot1)
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

    return dof_vel[1:, :].flatten()


def compute_dof_vels(local_rotation, fps=100):
    num_frames = local_rotation.shape[0]
    dt = 1.0 / fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = local_rotation[f]
        local_rot1 = local_rotation[f + 1]
        frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

    return dof_vels

    
class MotionLibSMPLGrip(MotionLibSMPL):  
    mesh_parsers = {
        0: SMPL_Parser(model_path="data/smpl", gender="neutral"),
        1: SMPL_Parser(model_path="data/smpl", gender="male"),
        2: SMPL_Parser(model_path="data/smpl", gender="female")
    }
    mujoco_2_smpl = [SMPL_MUJOCO_NAMES.index(q) for q in SMPL_BONE_ORDER_NAMES if q in SMPL_MUJOCO_NAMES]
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    
    def __init__(self, motion_lib_cfg, gender_betas=None):
        self.subjects = motion_lib_cfg.subjects
        self.failed_keys = motion_lib_cfg.failed_keys
        self.load_imu = motion_lib_cfg.load_imu
        self.load_insole = motion_lib_cfg.load_insole
        self.load_slam = motion_lib_cfg.load_slam
        self.load_pose_p = motion_lib_cfg.load_pose_p
        self.load_joint_gp_l = motion_lib_cfg.load_joint_gp_l
        self.load_joint_gp_g = motion_lib_cfg.load_joint_gp_g
        self.load_jvel_gp = motion_lib_cfg.load_jvel_gp
        self.load_angv_gp = motion_lib_cfg.load_angv_gp
        self.load_rvel_gp = motion_lib_cfg.load_rvel_gp
        self.load_trans_gp = motion_lib_cfg.load_trans_gp
        self.load_imu_vel = motion_lib_cfg.load_imu_vel
        self.load_imu_angv = motion_lib_cfg.load_imu_angv
        self.load_joint_gt_g = motion_lib_cfg.load_joint_gt_g
        self.load_object = motion_lib_cfg.load_object
        self.max_len = motion_lib_cfg.max_len
        self.win_len = motion_lib_cfg.win_len
        self.gender_betas = gender_betas

        global load_imu
        global load_insole
        global load_slam
        global load_pose_p
        global load_joint_gp_l
        global load_joint_gp_g
        global load_jvel_gp
        global load_angv_gp
        global load_rvel_gp
        global load_trans_gp
        global load_imu_vel
        global load_imu_angv
        global load_joint_gt_g
        global load_object

        load_imu = self.load_imu
        load_insole = self.load_insole
        load_slam = self.load_slam
        load_pose_p = self.load_pose_p
        load_joint_gp_l = self.load_joint_gp_l
        load_joint_gp_g = self.load_joint_gp_g
        load_jvel_gp = self.load_jvel_gp
        load_angv_gp = self.load_angv_gp
        load_rvel_gp = self.load_rvel_gp
        load_trans_gp = self.load_trans_gp
        load_imu_vel = self.load_imu_vel
        load_imu_angv = self.load_imu_angv
        load_joint_gt_g = self.load_joint_gt_g
        load_object = self.load_object

        super().__init__(motion_lib_cfg = motion_lib_cfg)

        return

    
    def load_data(self, motion_file, min_length=-1, im_eval=False):
        super().load_data(motion_file, min_length, im_eval, subjects=self.subjects)

        if self.failed_keys is not None:
            failed_keys = sorted(list(joblib.load(self.failed_keys)))
            print("Failed keys number: ", len(failed_keys))
            print(f"Failed keys: {failed_keys}")
        else:
            failed_keys = None

        print("Deviding motion data...", end="")
        _new_motion_data_keys = []
        _new_motion_data_list = []

        # global trans_diff
        # trans_diff = {}

        # print(f'Number of motion files: {len(self._motion_data_keys)}')

        for key, value in zip(self._motion_data_keys, self._motion_data_list):
            data_len = value['pose_quat_global'].shape[0]
            # trans = value['root_trans_offset'].clone()[:30].double()
            # pose_aa = to_torch(value['pose_aa'][:30])
            # trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, self.gender_betas[0], MotionLibSMPLGrip.mesh_parsers, fix_height_mode=FixHeightMode.ankle_fix)
            # if 'subj006_take024' in key or 'subj006_take025' in key:
            #     trans_fix -= 0.03
            # trans_diff[key] = trans_fix
            # value['seq_name'] = key

            win_len = self.win_len - 1
            if data_len <= self.max_len + win_len or self.max_len == -1:
                if failed_keys is None or key in failed_keys:
                    _new_motion_data_keys.append(key)
                    _new_motion_data_list.append(value)
            else:
                # start = random.randint(0, data_len - self.max_len - win_len)
                # end = start + self.max_len + win_len
                # new_key = key
                # new_value = {}

                # for k, v in value.items():
                #     if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                #         new_value[k] = v[start: end]
                #     elif isinstance(v, dict):
                #         if k == "objects":
                #             new_value[k] = v
                #         else:
                #             new_sub_value = {}
                #             for sub_k, sub_v in v.items():
                #                 new_sub_value[sub_k] = sub_v[start: end]
                #             new_value[k] = new_sub_value
                #     else:
                #         new_value[k] = v

                # if failed_keys is None or new_key in failed_keys:
                #     _new_motion_data_keys.append(new_key)
                #     _new_motion_data_list.append(new_value)


                for i in range((data_len - win_len) // self.max_len):
                    new_key = key + "_{:03d}".format(i)
                    new_value = {}
                    # if (not flags.im_eval) and (not flags.test):
                    if not flags.test:
                        start = random.randint(0, data_len - self.max_len - win_len)
                        end = start + self.max_len + win_len
                    else:
                        start = i * self.max_len
                        end = start + self.max_len + win_len
                    for k, v in value.items():
                        if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                            new_value[k] = v[start: end]
                        elif isinstance(v, dict):
                            if k == "objects":
                                new_value[k] = v
                            else:
                                new_sub_value = {}
                                for sub_k, sub_v in v.items():
                                    new_sub_value[sub_k] = sub_v[start: end]
                                new_value[k] = new_sub_value
                        else:
                            new_value[k] = v
                
                    if failed_keys is None or new_key in failed_keys:
                        _new_motion_data_keys.append(new_key)
                        _new_motion_data_list.append(new_value)
                    
                    ## GlobalPose Prediction
                    # data_path = f'data/PRISM/GlobalPose/{new_key}.npz'
                    # try:
                    #     data = np.load(data_path)
                    # except:
                    #     print(f"Failed to load {data_path}")
                    #     continue
                    # new_value['pose_p'] = data['pose_p']
                    # new_value['tran_gp'] =  data['tran_p']




        self._motion_data_keys = np.array(_new_motion_data_keys)
        self._motion_data_list = np.array(_new_motion_data_list)
        self._num_unique_motions = len(self._motion_data_list)
        print("Number of unique motions: ", self._num_unique_motions)
        print("Done.")

        # for key, value in zip(self._motion_data_keys, self._motion_data_list):
        #     data_len = value['pose_quat_global'].shape[0]
        #     trans = value['root_trans_offset'].clone()[:30].double()
        #     pose_aa = to_torch(value['pose_aa'][:30])
        #     trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, self.gender_betas[0], MotionLibSMPLGrip.mesh_parsers, fix_height_mode=FixHeightMode.ankle_fix)
        #     if 'subj006_take024' in key or 'subj006_take025' in key:
        #         trans_fix -= 0.03
        #     trans_diff[key] = trans_fix
        #     value['seq_name'] = key

        #     if data_len < self.max_len + self.win_len or self.max_len == -1:
        #         new_key = key + "_000"

        #         if failed_keys is None or new_key in failed_keys:
        #             _new_motion_data_keys.append(new_key)
        #             _new_motion_data_list.append(value)
        #     else:
        #         for i in range((data_len - self.win_len) // self.max_len):
        #             new_key = key + "_{:03d}".format(i)
        #             new_value = {}
        #             for k, v in value.items():
        #                 if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
        #                     new_value[k] = v[i * self.max_len: (i + 1) * self.max_len + self.win_len]
        #                 elif isinstance(v, dict):
        #                     if k == "objects":
        #                         new_value[k] = v
        #                     else:
        #                         new_sub_value = {}
        #                         for sub_k, sub_v in v.items():
        #                             new_sub_value[sub_k] = sub_v[i * self.max_len: (i + 1) * self.max_len + self.win_len]
        #                         new_value[k] = new_sub_value
        #                 else:
        #                     new_value[k] = v
                    
        #             ## GlobalPose Prediction
        #             # data_path = f'data/PRISM/GlobalPose/{new_key}.npz'
        #             # try:
        #             #     data = np.load(data_path)
        #             # except:
        #             #     print(f"Failed to load {data_path}")
        #             #     continue
        #             # new_value['pose_p'] = data['pose_p']
        #             # new_value['tran_gp'] =  data['tran_p']

        #             if failed_keys is None or new_key in failed_keys:
        #                 _new_motion_data_keys.append(new_key)
        #                 _new_motion_data_list.append(new_value)
        # self._motion_data_keys = np.array(_new_motion_data_keys)
        # self._motion_data_list = np.array(_new_motion_data_list)
        # self._num_unique_motions = len(self._motion_data_list)
        # print("Number of unique motions: ", self._num_unique_motions)
        # print("Done.")
    

    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1, augment_images = False):
        # if 'imu' in self.__dict__:
        #     self.imu = self.imu.to("cpu")
        #     self.insole = self.insole.to("cpu")
        #     del self.imu, self.insole
        torch.cuda.empty_cache()
        gc.collect()
        
        motions = super().load_motions(skeleton_trees=skeleton_trees, gender_betas=gender_betas, limb_weights=limb_weights, random_sample=random_sample, start_idx=start_idx, max_len=max_len)
        
        # self.imu, self.insole, self.box_large_pos, self.box_large_rotation, self.box_small_pos, self.box_small_rotation, self.chair_pos, self.chair_rotation = None, None, None, None, None, None, None, None
        # del self.imu, self.insole, self.box_large_pos, self.box_large_rotation, self.box_small_pos, self.box_small_rotation, self.chair_pos, self.chair_rotation
        # torch.cuda.empty_cache()

        self.indices = torch.cat([m.indices for m in motions], dim=0).to(self._device)  # (B*F, 2)
        self.imu = torch.stack([m.imu for m in motions]).to(self._device) if self.load_imu else None                       # (B, F+W, I, 18)
        self.insole = torch.stack([m.insole for m in motions]).to(self._device) if self.load_insole else None              # (B, F+W, 2, 22)
        self.slam = torch.stack([m.slam for m in motions]).to(self._device) if self.load_slam else None                    # (B, F+W, 3)
        self.pose_p = torch.stack([m.pose_p for m in motions]).to(self._device) if self.load_pose_p else None           # (B, F+W, 24, 4) - quaternion
        self.dof_pos_p = torch.stack([m.dof_pos_p for m in motions]).to(self._device) if self.load_pose_p else None     # (B, F+W, 69) - axis-angle
        self.dof_vel_p = torch.stack([m.dof_vel_p for m in motions]).to(self._device) if self.load_pose_p else None     # (B, F+W, 69) - axis-angle
        self.joint_p_l = torch.stack([m.joint_p_l for m in motions]).to(self._device) if self.load_joint_gp_l else None  # (B, F+W, 24, 3)
        self.joint_p_g = torch.stack([m.joint_p_g for m in motions]).to(self._device) if self.load_joint_gp_g else None  # (B, F+W, 24, 3)
        self.jvel_gp = torch.stack([m.jvel_gp for m in motions]).to(self._device) if self.load_jvel_gp else None           # (B, F+W, 24, 3)
        self.angv_gp = torch.stack([m.angv_gp for m in motions]).to(self._device) if self.load_angv_gp else None           # (B, F+W, 6, 3)
        self.rvel_gp = torch.stack([m.rvel_gp for m in motions]).to(self._device) if self.load_rvel_gp else None           # (B, F+W, 6, 3)
        self.trans_gp = torch.stack([m.trans_gp for m in motions]).to(self._device) if self.load_trans_gp else None        # (B, F+W, 3)
        self.imu_vel = torch.stack([m.vel for m in motions]).to(self._device) if self.load_imu_vel else None             # (B, F+W, I, 3)
        self.imu_angv = torch.stack([m.angv for m in motions]).to(self._device) if self.load_imu_angv else None           # (B, F+W, I, 3)
        self.joint_gt_g = torch.stack([m.joint_gt_g for m in motions]).to(self._device) if self.load_joint_gt_g else None  # (B, F+W, 24, 3)

        if self.load_object:
            # tran_diff = torch.tensor([-0.2109 , -0.19225,  0.     ]).to(self._device)
            # tran_diff = tran_diff.repeat(len(motions), 1)
            self.box_large_pos = torch.from_numpy(np.array([m.objects['BoxLarge']['center'] for m in motions])).float().to(self._device) #+ tran_diff
            self.box_large_rotation = torch.from_numpy(np.array([m.objects['BoxLarge']['rotation'] for m in motions])).float().to(self._device)
            self.box_large_vertices = torch.from_numpy(np.array([m.objects['BoxLarge']['vertices'] for m in motions])).float().to(self._device) #+ tran_diff
            self.box_small_pos = torch.from_numpy(np.array([m.objects['BoxSmall']['center'] for m in motions])).float().to(self._device) #+ tran_diff
            self.box_small_rotation = torch.from_numpy(np.array([m.objects['BoxSmall']['rotation'] for m in motions])).float().to(self._device)
            self.box_small_vertices = torch.from_numpy(np.array([m.objects['BoxSmall']['vertices'] for m in motions])).float().to(self._device) #+ tran_diff
            self.chair_pos = torch.from_numpy(np.array([m.objects['Chair']['center'] for m in motions])).float().to(self._device) #+ tran_diff
            self.chair_rotation = torch.from_numpy(np.array([m.objects['Chair']['rotation'] for m in motions])).float().to(self._device)
            self.chair_vertices = torch.from_numpy(np.array([m.objects['Chair']['vertices'] for m in motions])).float().to(self._device) #+ tran_diff

        return motions

    @staticmethod
    def fix_trans_height(contact,
                        box_corners,
                        foot_pos_gt,
                        pose_aa,
                        trans,
                        curr_gender_betas,
                        mesh_parsers,
                        fix_height_mode):
        """
        box_corners: (M, 4, 3)
        - M: number of boxes in the scene
        - 4: number of corners of the quadrilateral (box top face)
        - 3: (x, y, z)
        contact: (F, 2, 1)
        - F: number of frames
        - contact[:, 0, 0]: left toe contact flag
        - contact[:, 1, 0]: right toe contact flag
        """

        # ========== Helper functions ========== #
        def points_in_convex_quadrilateral(
            points_xy: torch.Tensor,  # (F, N, 2)
            quad_xy: torch.Tensor     # (F, 4, 2)
        ) -> torch.Tensor:
            """
            For each frame, return a bool mask indicating which of the N 2D points lie
            inside the convex quadrilateral quad_xy. Output shape: (F, N).
            """
            F, N, _ = points_xy.shape
            # quad_xy: (F, 4, 2)
            # points_xy: (F, N, 2)

            quad_xy_ex = quad_xy.unsqueeze(1).expand(F, N, 4, 2)     # (F, N, 4, 2)
            points_xy_ex = points_xy.unsqueeze(2).expand(F, N, 4, 2) # (F, N, 4, 2)

            # Shift the quadrilateral vertices by one and form edges from adjacent pairs
            quad_shifted = torch.roll(quad_xy_ex, shifts=-1, dims=2)  # (F, N, 4, 2)
            edges = quad_shifted - quad_xy_ex  # (F, N, 4, 2)

            def cross_2d(u, v):
                # z component of the 2D cross product of u and v
                return u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]

            vec_to_points = points_xy_ex - quad_xy_ex  # (F, N, 4, 2)
            cross_vals = cross_2d(edges, vec_to_points)  # (F, N, 4)

            cond_pos = (cross_vals >= 0).all(dim=-1)  # (F, N)
            cond_neg = (cross_vals <= 0).all(dim=-1)  # (F, N)
            inside_mask = cond_pos | cond_neg  # True means inside
            return inside_mask

        def compute_box_or_ground_z(toe_xy: torch.Tensor,
                                    box_corners: torch.Tensor) -> torch.Tensor:
            """
            toe_xy: (F, 2)        Per-frame toe (x, y).
            box_corners: (M, 4, 3) Static box corner positions (x, y, z).

            Returns: (F,)
            - For each frame, returns the top-face z of the box the toe rests on
              (taking the highest box if multiple).
            - Returns 0.0 (ground) if no box contains the toe.
            """
            F = toe_xy.shape[0]
            M = box_corners.shape[0]

            final_replace_z = torch.zeros((F,), device=toe_xy.device, dtype=toe_xy.dtype)
            # (F, 1, 2)   shape adjusted for points_in_convex_quadrilateral
            toe_xy_ex = toe_xy.unsqueeze(1)  # (F, 1, 2)

            for m in range(M):
                # Corners (4, 3) of the m-th box
                box_m = box_corners[m]     # shape: (4, 3)
                box_xy = box_m[:, :2]      # shape: (4, 2)
                box_xy_ex = box_xy.unsqueeze(0).expand(F, 4, 2)  # (F,4,2)

                # Top-face z taken as the mean of the 4 vertex z values (could also use min/max)
                box_z = box_m[:, 2].mean()  # scalar
                box_z_expanded = torch.full((F,), box_z, device=toe_xy.device, dtype=toe_xy.dtype)

                # inside_mask: (F, 1)
                inside_mask = points_in_convex_quadrilateral(toe_xy_ex, box_xy_ex)
                inside_mask = inside_mask.squeeze(-1)  # (F,)

                # box_z for inside frames, 0 otherwise
                candidate = torch.where(inside_mask, box_z_expanded, torch.zeros_like(box_z_expanded))

                # Keep the highest box z so far
                final_replace_z = torch.max(final_replace_z, candidate)

            return final_replace_z

        # ========== Main processing ========== #
        if fix_height_mode == FixHeightMode.no_fix or fix_height_mode == FixHeightMode.full_fix:
            raise NotImplementedError

        frame_check = 300
        
        with torch.no_grad():
            # (F,) bool
            contact = contact[:frame_check].bool()
            contact_frames_left = np.logical_and(contact[:, 0, 0], contact[:, 0, 1])  # left foot contact
            contact_frames_right = np.logical_and(contact[:, 1, 0], contact[:, 1, 1])  # right foot contact

            gender = curr_gender_betas[0]
            betas = curr_gender_betas[1:]
            mesh_parser = mesh_parsers[gender.item()]

            # Vertex / joint positions: (F, #verts, 3), (F, #joints, 3)
            vertices_curr, joints_curr = mesh_parser.get_joints_verts(
                pose_aa[:frame_check], betas[None, :], trans[:frame_check]
            )

            # Offset between SMPL root (joint 0) and trans, shape (F, 3)
            offset = joints_curr[:, 0] - trans[:frame_check]

            # Indices for L_Toe / R_Toe
            L_Toe_idx = mesh_parser.joint_names.index("L_Toe")
            R_Toe_idx = mesh_parser.joint_names.index("R_Toe")
            L_Ankle_idx = mesh_parser.joint_names.index("L_Ankle")
            R_Ankle_idx = mesh_parser.joint_names.index("R_Ankle")

            # Toe world coords = joints_curr[:, toe_idx] - offset
            L_Toe_world = joints_curr[:, L_Toe_idx] - offset
            R_Toe_world = joints_curr[:, R_Toe_idx] - offset
            L_Ankle_world = joints_curr[:, L_Ankle_idx] - offset
            R_Ankle_world = joints_curr[:, R_Ankle_idx] - offset

            L_pos_gt = foot_pos_gt[:frame_check, 0]  # (F, 3)
            R_pos_gt = foot_pos_gt[:frame_check, 1]  # (F, 3)

            # Get the box or ground z, shape (F,)
            L_box_or_ground_z = compute_box_or_ground_z(L_pos_gt[:, :2], box_corners)
            R_box_or_ground_z = compute_box_or_ground_z(L_pos_gt[:, :2], box_corners)

            # if detect non-zero value in L_box_or_ground_z and R_box_or_ground_z, then use the average of them as the height tolerance
            # if (L_box_or_ground_z != 0).any() and (R_box_or_ground_z != 0).any():
            #     print("L_box_or_ground_z", L_box_or_ground_z)
            #     print("R_box_or_ground_z", R_box_or_ground_z)

            # Actual toe height = (toe z) - (box or ground z)
            L_Toe_heights = L_Toe_world[:, 2] - L_box_or_ground_z
            R_Toe_heights = R_Toe_world[:, 2] - R_box_or_ground_z
            L_Ankle_heights = L_Ankle_world[:, 2] - L_box_or_ground_z
            R_Ankle_heights = R_Ankle_world[:, 2] - R_box_or_ground_z
            L_Foot_heights = np.minimum(L_Toe_heights, L_Ankle_heights)
            R_Foot_heights = np.minimum(R_Toe_heights, R_Ankle_heights)

            print(f'L_Toe: Max {L_Foot_heights.max()}, Min {L_Foot_heights.min()}, Mean {L_Foot_heights.mean()}')
            print(f'R_Toe: Max {R_Foot_heights.max()}, Min {R_Foot_heights.min()}, Mean {R_Foot_heights.mean()}')

            # Keep only contacting frames and average
            L_valid_heights = L_Toe_heights[contact_frames_left]  # (n_frames,)
            R_valid_heights = R_Toe_heights[contact_frames_right] # (n_frames,)

            if len(L_valid_heights) > 0:
                avg_left_height = L_valid_heights.mean()
            else:
                avg_left_height = None

            if len(R_valid_heights) > 0:
                avg_right_height = R_valid_heights.mean()
            else:
                avg_right_height = None

            # Average the left and right means
            if avg_left_height is not None and avg_right_height is not None:
                avg_height = 0.5 * (avg_left_height + avg_right_height)
            elif avg_left_height is not None:
                avg_height = avg_left_height
            elif avg_right_height is not None:
                avg_height = avg_right_height
            else:
                avg_height = 0.0  # no contact frames on either side

            # Compute diff_fix and adjust the height
            height_tolerance = 0.07
            diff_fix = avg_height - height_tolerance
            trans[..., -1] -= diff_fix

            return trans, diff_fix
    
    
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, shape_params, mesh_parsers, config, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        max_len = config.max_length
        win_len = config.win_len
        fix_height = config.fix_height
        res = {}
        for f in range(len(motion_data_list)):
            assert (len(ids) == len(motion_data_list))
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]

            # if not isinstance(curr_file, dict) and osp.isfile(curr_file):
            #     key = motion_data_list[f].split("/")[-1].split(".")[0]
            #     curr_file = joblib.load(curr_file)[key]
            # curr_file = joblib.load(curr_file)
            curr_gender_beta = shape_params[f]
            seq_len = curr_file['root_trans_offset'].shape[0]

            # if max_len == -1 or seq_len < max_len + win_len:
            #     start, end = 0, seq_len - win_len
            # else:
            #     start = random.randint(0, seq_len - max_len - win_len)
            #     end = start + max_len
            start = 0
            end = seq_len - win_len
            # start = 400
            # end = 1400

            fps = curr_file.get("fps", 100)
            F = end - start
            W = win_len
            I = 6  # 6 IMUs

            ## Load motion data
            trans = curr_file['root_trans_offset'].clone()[start:end].double()
            pose_aa = to_torch(curr_file['pose_aa'][start:end])
            pose_quat_global = to_torch(curr_file['pose_quat_global'][start:end])
            tran_orig = to_torch(curr_file['trans_orig'][start:end])   
            # trans[..., -1] += 0.03         
            # trans[..., -1] -= trans_diff[curr_file['seq_name']]
            # trans_diff_xy = tran_orig[..., :2] - trans[..., :2]
            # trans[..., :2] += trans_diff_xy
            sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees[f], pose_quat_global, trans, is_local=False)

            # sk_jpos = sk_state.global_translation
            # np.savez('output/sk_jpos.npz', jpos=sk_jpos.cpu().numpy())

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, fps)
            curr_dof_vels = compute_motion_dof_vels(curr_motion)


            if load_imu:
                imu_acc = curr_file['imu_data']['acc'][start:end + win_len]  # (F+W, I, 3)
                imu_ori = curr_file['imu_data']['ori'][start:end + win_len]  # (F+W, I, 3, 3)

                # imu_acc_gt = curr_file['imu_data']['gt_acc'][start:end + win_len]  # (F+W, I, 3)
                # imu_ori_gt = curr_file['imu_data']['gt_ori'][start:end + win_len]  # (F+W, I, 3, 3)
                # imu_vel = curr_file['imu_data']['gt_vel'][start:end + win_len]  # (F+W, I, 3)
                # imu_pos_gt = curr_file['imu_data']['gt_pos'][start:end + win_len]  # (F+W, I, 3)
                # foot_pos_gt = imu_pos_gt[:, 0:2, :]
                # foot_pos_gt = torch.cat([foot_pos_gt, foot_pos_gt, foot_pos_gt], axis=1) # (F+W, 6, 3)

                # # Alighn IMU orientation and acceleration to initial foot pose
                # init_imu_ori_gt = imu_ori_gt[0]
                # foot_imu_acc = imu_acc[:, 0:2, :] # (F+W, 2, 3, 3)
                # foot_imu_ori = imu_ori[:, 0:2, :] # (F+W, 2, 3, 3)
                # for i, side in enumerate(['left', 'right']):
                #     init_foot_imu_ori_inv = torch.inverse(foot_imu_ori[0, i])
                #     foot_imu_ori[:, i] = init_imu_ori_gt[i] @ init_foot_imu_ori_inv @ foot_imu_ori[:, i]
                #     foot_imu_acc[:, i] = foot_imu_acc[:, i] @ init_foot_imu_ori_inv.T @ init_imu_ori_gt[i].T
                # imu_ori[:, 0:2, :] = foot_imu_ori
                # imu_acc[:, 0:2, :] = foot_imu_acc
                
                # imu_ori_gt = imu_ori_gt.reshape(F+W, I, -1)  # (F+W, I, 9)
                # imu = torch.cat([imu_acc_gt, imu_ori_gt], dim=-1)  # (F+W, I, 12)  (3 + 9 = 12)
                # imu = torch.cat([imu_acc_gt, imu_ori_gt, imu_vel, foot_pos_gt], dim=-1) # (F+W, I, 18)  (3 + 9 + 3 + 3 = 18)

                # imu_ori = imu_ori.reshape(F+W, I, -1)  # (W, I, 9)
                # imu = torch.cat([imu_acc, imu_ori], dim=-1)  # (W, I, 12)  (3 + 9 = 12)
                # curr_motion.imu = imu  # (F+W, I, 18)
                R_h_i = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float()
                R_i_h = torch.inverse(R_h_i)
                imu_quat = quat_from_rotation_matrix(imu_ori @ R_i_h)  # (B, T, 6, 4)  # Humanoid Coordinate
                imu = torch.cat([imu_acc, imu_quat], dim=-1)  # (F+W, I, 7)  (3 + 4 = 7)
                curr_motion.imu = imu  # (F+W, I, 7)


            if load_insole:
                curr_motion.insole = curr_file['insole_data'][start:end + win_len]  # (F + win_len, 2, 5)

            if load_slam:
                # curr_motion.slam = curr_file['slam_data'][start:end + win_len]  # (F+W, 3)
                # TODO: fix humanoid height to use slam data
                curr_motion.slam = torch.from_numpy(curr_file['joints_smpl'][start:end + win_len, 15]).float()  # (F+W, 3)


            if load_pose_p:
                # curr_motion.pose_p = curr_file['poses_p'][start:end + win_len]  # (F+W, 24, 3, 3)

                def local_to_global(local_oris, parents):
                    _, n_joints, _, _ = local_oris.shape
                    global_oris = torch.zeros_like(local_oris)

                    for j in range(n_joints):
                        if parents[j] < 0: # root rotation
                            global_oris[..., j, :, :] = local_oris[..., j, :, :]
                        else:
                            parent_rot = global_oris[..., parents[j], :, :]
                            local_rot = local_oris[..., j, :, :]
                            global_oris[..., j, :, :] = torch.matmul(parent_rot, local_rot)

                    res = global_oris.reshape((-1, n_joints, 3, 3))
                    return res

                pose_p_aa = curr_file['pose_p'][start:end + win_len]  # (F+W, 24, 3)
                pose_p_mat = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(pose_p_aa).float())
                pose_p_mat = pose_p_mat.reshape(F+W, 24, 3, 3)
                pose_p_mat = local_to_global(pose_p_mat, MotionLibSMPLGrip.mesh_parsers[0].parents)  # (F+W, 24, 3, 3)
                pose_p_mat = pose_p_mat[:, MotionLibSMPLGrip.smpl_2_mujoco]
                R_h_i = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float()
                R_i_h = torch.inverse(R_h_i)
                pose_p_quat = quat_from_rotation_matrix(pose_p_mat @ R_i_h) # (F+W, 24, 4)
                curr_motion.pose_p = pose_p_quat  # (F+W, 24, 4)
                
                # Also store local axis-angle (used for DOF position computation)
                dof_pos_aa = pose_aa.reshape(-1, 24, 3)[:, MotionLibSMPLGrip.smpl_2_mujoco][:, :, [2, 0, 1]] # (F+W, 24, 3)
                dof_pos_quat = torch_utils.exp_map_to_quat(dof_pos_aa.reshape(-1, 3)).reshape(-1, 24, 4) # (F+W, 24, 4)
                dof_vel_pred = compute_dof_vels(dof_pos_quat)
                curr_motion.dof_pos_p = dof_pos_aa[:, 1:].reshape(-1, 69)  # (F+W, 69)
                curr_motion.dof_vel_p = dof_vel_pred.reshape(-1, 69)  # (F+W, 69)
                # print(f'dof_pos_pred shape: {curr_motion.dof_pos_pred.shape}')
                # print(f'dof_vel_pred shape: {curr_motion.dof_vel_pred.shape}')
                # print(f'dof_pos_pred: {curr_motion.dof_pos_pred.cpu().numpy()[0]}')
                # print(f'dof_vel_pred: {curr_motion.dof_vel_pred.cpu().numpy()[0]}')

            # if load_joint_gp_l:
            #     curr_motion.joint_p_l = curr_file['joints_p'][start:end + win_len]  # (F+W, 24, 3)

            if load_joint_gp_g:
                joint_p_g = torch.from_numpy(curr_file['joints_p'][start:end + win_len]).float()  # (F+W, 24, 3)
                curr_motion.joint_p_g = joint_p_g[:, MotionLibSMPLGrip.smpl_2_mujoco]

                # joints_gp_l = curr_file['joints_gp'][start:end + win_len]  # (F+W, 24, 3)
                # root_ori = curr_file['poses_gp'][start:end + win_len, 0]  # (F+W, 3, 3)
                # joints_gp_g = joints_gp_l @ root_ori.transpose(1, 2)  # [F+W, 24, 3]
                # joints_gp_g = joints_gp_g[:, MotionLibSMPLGrip.smpl_2_mujoco].reshape(-1, 24, 3)  # (F+W, 24, 3)
                # curr_motion.joint_p_g = joints_gp_g

            # if load_jvel_gp:
            #     def calc_joint_vel(joints, dt=1/100):
            #         vel = torch.zeros_like(joints)
            #         vel[0] = (joints[1] - joints[0]) / dt
            #         vel[1:-1] = (joints[2:] - joints[:-2]) / (2 * dt)
            #         vel[-1] = (joints[-1] - joints[-2]) / dt
            #         return vel
            #     jvel_gp = calc_joint_vel(joints_gp_g, dt=1/fps)  # (F+W, 24, 3)
            #     curr_motion.jvel_gp = jvel_gp

            
            # if load_rvel_gp:
            #     curr_motion.rvel_gp = curr_file['vel_gp'][start:end + win_len]  # (F+W, 24, 3)

            # if load_trans_gp:
            #     curr_motion.trans_gp = curr_file['trans_gp'][start:end + win_len]  # (F+W, 3)

            if load_imu_vel:
                curr_motion.vel = torch.from_numpy(curr_file['imu_data']['vel_p'][start:end + win_len]).float()  # (F+W, I, 3)

            if load_imu_angv:
                imu_angv = SkeletonMotion._compute_angular_velocity(imu_quat, time_delta=0.01)  # (F+W, 6, 3)
                curr_motion.angv = imu_angv

            # if load_joint_gt_g:
            #     joint_gt_g = curr_file['joints_smpl'][start:end + win_len]  # (F+W, 24, 3)
            #     curr_motion.joint_gt_g = torch.from_numpy(joint_gt_g).float()

            if load_object:
                objects = curr_file['objects']
                curr_motion.objects = objects

            indices = []
            for i in range(F):
                indices.append([curr_id, i, i+win_len])
            indices = torch.tensor(indices)
            curr_motion.indices = indices


            curr_file = {'pose_aa': pose_aa}  # for memory saving
            curr_motion.dof_vels = curr_dof_vels
            curr_motion.gender_beta = curr_gender_beta
            res[curr_id] = (curr_file, curr_motion)

        if not queue is None:
            queue.put(res)
        else:
            return res

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * 100 / self._motion_fps).int()
        else:
            return (self._motion_num_frames[motion_ids] * 100 / self._motion_fps).int()
        
    
    def get_motion_state(self, motion_ids, motion_times, offset=None, load_sensors=False):
        return_dict = {}
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        
        if load_sensors:
            ############################## Sensors ##############################
            blend_idx = blend
            blend_idx = (blend_idx >= 0.5).long() # Can not interperlate. 
            f0 = torch.gather(torch.stack([f0l, f1l], dim = -1), 1, blend_idx).squeeze(1)
            indices = self.indices[f0]
            B = indices.shape[0]
            T = indices[0, 2] - indices[0, 1]
            starts = indices[:, 1]  # shape [B]
            offsets = torch.arange(T).unsqueeze(0).expand(B, T).to(self._device)
            gather_idx = starts.unsqueeze(1) + offsets
            batch_idx = torch.arange(B).unsqueeze(1).expand(B, T).to(self._device)

            load_items = ['imu', 'insole', 'slam', 'pose_p', 'dof_pos_p', 'dof_vel_p', 'joint_p_l', 'joint_p_g', 'jvel_gp', 'angv_gp', 'rvel_gp', 'trans_gp', 'imu_vel', 'imu_angv', 'joint_gt_g']
            for item in load_items:
                if self.__dict__[item] is not None:
                    return_dict[item] = self.__dict__[item][batch_idx, gather_idx]

                    # for idx in range(len(f0)):
                    #     indices = self.indices[f0[idx]]
                    #     curr_id = indices[0].item()
                    #     print(f'Indices: {indices}, curr_id: {curr_id}')
                    #     start_idx = indices[1].item()
                    #     end_idx = indices[2].item()
                    #     data_list.append(self.__dict__[item][curr_id][start_idx:end_idx])
                    # return_dict[item] = torch.stack(data_list, dim=0)
        

        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "local_rot": local_rot.clone(), 
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        })
        
        return return_dict
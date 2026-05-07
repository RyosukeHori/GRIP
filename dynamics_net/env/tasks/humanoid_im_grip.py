"""HumanoidImGrip: Isaac Gym task that drives DynamicsNet on top of
KinematicsNet predictions, with optional test-time fall recovery.

The class extends ``HumanoidIm`` with:
    * Per-IMU observation construction conditioned on ``num_imus``
    * A heightmap of static-object surfaces around the agent
    * Box / chair object actors loaded from the motion library
    * History buffers + a discriminator-gated fall-recovery routine that can
      restore the humanoid to a kinematics-only estimate when it falls
      (test-time, ``env.fall_recovery=True`` only)
"""
import gc
from typing import OrderedDict

import numpy as np
import torch
from easydict import EasyDict
from isaacgym import gymapi, gymtorch
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES

import dynamics_net.env.tasks.humanoid_im as humanoid_im
from dynamics_net.utils import torch_utils
from dynamics_net.utils.flags import flags
from dynamics_net.utils.motion_lib_base import FixHeightMode
from dynamics_net.utils.motion_lib_smpl_grip import MotionLibSMPLGrip
from dynamics_net.utils.torch_utils import *  # noqa: F401,F403  (legacy: many helpers used unqualified)
from poselib.poselib.core.rotation3d import rot_matrix_from_quaternion


# Static, room-scale obstacles registered with Isaac Gym. Sizes / centres mirror
# the per-take object metadata produced by ``data_process/dynamics_dataset.py``.
_OBJECT_SPECS = {
    'BoxLarge': {'size': np.array([0.2840, 0.7438, 0.1958]), 'center': np.array([-5.0, 0.0, -1.0])},
    'BoxSmall': {'size': np.array([0.2744, 0.3857, 0.1903]), 'center': np.array([-5.0, 0.0, -1.0])},
    'Chair':    {'size': np.array([0.2457, 0.2473, 0.4442]), 'center': np.array([-4.0, 0.0, -1.0])},
}

# Number of frames retained by the fall-recovery history buffers (1 s @ 100 Hz).
_HIST_LEN = 100
# Fall is declared when root z < this threshold AND discriminator probability < 0.7.
_FALL_ROOT_Z = 0.30
_FALL_DISC_PROB = 0.7


class HumanoidImGrip(humanoid_im.HumanoidIm):

    # --------------------------------------------------------------------- #
    # Initialization                                                        #
    # --------------------------------------------------------------------- #

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._read_env_config(cfg['env'], headless=cfg['headless'])

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        self._init_body_indices()
        self._init_state_buffers()
        self._init_discriminator_slots()

    def _read_env_config(self, env_cfg: dict, headless: bool) -> None:
        """Cache every ``env.*`` flag that the rest of the class reads."""
        self.num_imus = env_cfg.get("num_imus", 4)
        self.win_len = env_cfg.get("win_len", 50)
        self.rand_seq = env_cfg.get("rand_seq", False)
        self.latent_dim = env_cfg.get("latent_dim", 1024)

        load_cfg = env_cfg.get('load', {})
        self.load_imu = load_cfg.get("imu", True)
        self.load_insole = load_cfg.get("insole", True)
        self.load_slam = load_cfg.get("slam", True)
        self.load_pose_p = load_cfg.get("pose_p", True)
        self.load_joint_gp_l = load_cfg.get("joint_p_l", True)
        self.load_joint_gp_g = load_cfg.get("joint_p_g", True)
        self.load_jvel_gp = load_cfg.get("jvel_gp", True)
        self.load_angv_gp = load_cfg.get("angv_gp", True)
        self.load_rvel_gp = load_cfg.get("rvel_gp", True)
        self.load_trans_gp = load_cfg.get("trans_gp", True)
        self.load_imu_vel = load_cfg.get("imu_vel", True)
        self.load_imu_angv = load_cfg.get("imu_angv", True)
        self.load_joint_gt_g = load_cfg.get("joint_gt_g", True)
        self.load_object = load_cfg.get("object", True)

        self.refiner = env_cfg.get("refiner", False)
        self.joint_loss_weight = env_cfg.get("joint_loss_weight", 1.0)
        self.vel_loss_weight = env_cfg.get("vel_loss_weight", 1.0)
        self.pretrain = env_cfg.get("pretrain", False)
        self.fall_penalty_scale = env_cfg.get("fall_penalty", 5.0)

        # Fall-recovery is opt-in at test time only.
        self.fall_recovery = env_cfg.get("fall_recovery", False)

        if self.rand_seq and headless:
            print("!!!!!!! env.rand_seq can be True only when env.headless is False !!!!!!!!")
            print("!!!!!!! env.rand_seq is set to False !!!!!!!!")
            self.rand_seq = False

    def _init_body_indices(self) -> None:
        """Resolve named body / DOF indices once so the obs builders stay terse."""
        bn = self._body_names
        self.ankle_rb_idx = [bn.index(n) for n in ['L_Ankle', 'R_Ankle']]
        self.toe_rb_idx = [bn.index(n) for n in ['L_Toe', 'R_Toe']]
        self.ee_rb_idx = [bn.index(n) for n in ['L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist', 'Head', 'Pelvis']]
        self.ub_rb_idx = [bn.index(n) for n in ['Pelvis', 'Chest', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow']]
        self.ub_jnt_idx = [bn.index(n) for n in [
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine', 'Neck', 'Head',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
            'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand',
        ]]
        self.upper_imu_idx = [2, 3, 4, 5]

        self.mujoco_2_smpl = [SMPL_MUJOCO_NAMES.index(q) for q in SMPL_BONE_ORDER_NAMES if q in SMPL_MUJOCO_NAMES]
        self.smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]

        self.R_heading = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        self.pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5]) if self._has_upright_start else sRot.identity()

    def _init_state_buffers(self) -> None:
        """Allocate per-env GT / history / current-estimate buffers."""
        N = self.num_envs

        # Reference (ground-truth) buffers used for fall reset & rewards.
        self.ref_body_ang_vel = torch.zeros_like(self._rigid_body_ang_vel)
        self.ref_dof_vel = torch.zeros_like(self._dof_vel)

        self.prev_rigid_body_vel = torch.zeros_like(
            self._rigid_body_vel, device=self.device, dtype=torch.float)
        self.grf_ins = torch.zeros((N, 24, 3), device=self.device, dtype=torch.float)

        # Fall-recovery history (last _HIST_LEN frames).
        self.root_pos_hist_buffer = torch.zeros((N, _HIST_LEN, 3), device=self.device, dtype=torch.float)
        self.root_vel_hist_buffer = torch.zeros((N, _HIST_LEN, 3), device=self.device, dtype=torch.float)
        self.body_rot_hist_buffer = torch.zeros((N, _HIST_LEN, 24, 4), device=self.device, dtype=torch.float)
        self.body_pos_hist_buffer = torch.zeros((N, _HIST_LEN, 24, 3), device=self.device, dtype=torch.float)
        self.contact_hist_buffer = torch.zeros((N, _HIST_LEN, 2), device=self.device, dtype=torch.bool)

        # Latest KinematicsNet-derived estimates, written each step in _compute_task_obs.
        self.current_root_vel = torch.zeros((N, 3), device=self.device, dtype=torch.float)
        self.current_body_rot = torch.zeros((N, 24, 4), device=self.device, dtype=torch.float)
        self.current_body_rot[:, :, 3] = 1.0  # quaternion identity (w = 1)
        self.current_body_pos = torch.zeros((N, 24, 3), device=self.device, dtype=torch.float)
        self.current_dof_pos = torch.zeros((N, 69), device=self.device, dtype=torch.float)
        self.current_dof_vel = torch.zeros((N, 69), device=self.device, dtype=torch.float)
        self.current_contact = torch.zeros((N, 2), device=self.device, dtype=torch.bool)

        self.total_num_falls = torch.zeros((1,), device=self.device, dtype=torch.int32)

    def _init_discriminator_slots(self) -> None:
        """Set by ``set_discriminator`` from the player at test time."""
        self.disc_model = None
        self.amp_input_mean_std = None

    def set_discriminator(self, disc_model, amp_input_mean_std=None) -> None:
        """Inject the AMP discriminator (called from the rl_games player)."""
        self.disc_model = disc_model
        self.amp_input_mean_std = amp_input_mean_std

    # --------------------------------------------------------------------- #
    # Scene / asset construction                                            #
    # --------------------------------------------------------------------- #

    def _create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._heightmap_handles = [[] for _ in range(num_envs)]
            self._load_heightmap_asset()

        super()._create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_scene(env_id, env_ptr)

    def _build_scene(self, env_id: int, env_ptr) -> None:
        """Add the heightmap markers (when not headless) and the static objects."""
        self._square_grid = self._make_square_grid()
        self._num_heightmap_points = self._square_grid.shape[0]

        if not self.headless:
            for _ in range(self._num_heightmap_points):
                handle = self.gym.create_actor(
                    env_ptr, self._heightmap_asset, gymapi.Transform(),
                    "heightmap", self.num_envs + 10, 1, 0,
                )
                self.gym.set_rigid_body_color(env_ptr, handle, 0,
                                              gymapi.MESH_VISUAL,
                                              gymapi.Vec3(0.0, 0.8, 0.0))
                self._heightmap_handles[env_id].append(handle)

        # Static objects (boxes / chair) — fixed-base, gravity-disabled.
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.density = 1000.0           # ignored when fix_base_link=True; defensive
        asset_options.disable_gravity = True
        for name, spec in _OBJECT_SPECS.items():
            box_asset = self.gym.create_box(self.sim, *spec['size'], asset_options)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*spec['center'])
            self.gym.create_actor(env_ptr, box_asset, pose, name, env_id, 1)

    def _make_square_grid(self, num_points: int = 25, size: float = 1.5) -> torch.Tensor:
        x = np.linspace(-size / 2, size / 2, num_points)
        y = np.linspace(-size / 2, size / 2, num_points)
        xx, yy = np.meshgrid(x, y)
        local_grid = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)
        return torch.tensor(local_grid).to(self.device)

    def _load_heightmap_asset(self) -> None:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.density = 0.0
        asset_options.disable_gravity = True
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.override_inertia = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._heightmap_asset = self.gym.load_asset(
            self.sim, "dynamics_net/data/assets/urdf/", "traj_marker_small.urdf", asset_options,
        )

    # --------------------------------------------------------------------- #
    # Heightmap                                                             #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _yaw_rotmat(root_rot: torch.Tensor) -> np.ndarray:
        """[B, 3, 3] heading rotation derived from the root orientation."""
        x_axis = root_rot[:, :, 0]
        yaw = np.arctan2(x_axis[:, 1], x_axis[:, 0])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        zeros, ones = np.zeros_like(yaw), np.ones_like(yaw)
        return np.stack([
            np.stack([cos_y, -sin_y, zeros], axis=1),
            np.stack([sin_y,  cos_y, zeros], axis=1),
            np.stack([zeros,  zeros, ones],  axis=1),
        ], axis=1)

    def _update_heightmap(self) -> None:
        root_pos = self._rigid_body_pos[:, 0]
        root_rot = rot_matrix_from_quaternion(self._rigid_body_rot[:, 0])
        B = root_pos.shape[0]

        # Yaw-aligned, mirrored heading frame (used by the rest of the obs builder too).
        R_heading = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ self._yaw_rotmat(root_rot)

        # Place the local grid in the heading frame, anchored at the root.
        grid_local = self._square_grid.unsqueeze(0).expand(B, -1, -1)
        grid_world = np.einsum('fij,fnj->fni', R_heading, grid_local)
        grid_world = torch.from_numpy(grid_world).float().cuda()
        grid_world += root_pos.unsqueeze(1)
        grid_world[:, :, 2] = 0.01

        if self.load_object:
            grid_world = self._project_grid_onto_objects(grid_world)

        self._heightmap_pos[:] = grid_world
        self.R_heading[:] = torch.from_numpy(R_heading).float().cuda()
        if not self.headless:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(self._heightmap_actor_ids),
                len(self._heightmap_actor_ids),
            )

    def _project_grid_onto_objects(self, grid_world: torch.Tensor) -> torch.Tensor:
        """Lift heightmap points onto static-object top surfaces if they fall inside."""
        objs = torch.stack(
            [self.box_large_vertices, self.box_small_vertices, self.chair_vertices], dim=1)
        x_min = objs[..., 0].min(dim=-1)[0][:, :, None]
        x_max = objs[..., 0].max(dim=-1)[0][:, :, None]
        y_min = objs[..., 1].min(dim=-1)[0][:, :, None]
        y_max = objs[..., 1].max(dim=-1)[0][:, :, None]
        z_top = objs[..., 2].max(dim=-1)[0][:, :, None]

        gx = grid_world[:, :, 0][:, None, :]
        gy = grid_world[:, :, 1][:, None, :]
        gz = grid_world[:, :, 2][:, None, :]

        inside = (gx >= x_min) & (gx <= x_max) & (gy >= y_min) & (gy <= y_max) & (gz <= z_top)
        candidate = z_top + 0.01
        zeros = torch.tensor(0.0, device=grid_world.device, dtype=grid_world.dtype)
        z_per_obj = torch.where(inside, candidate, zeros)
        z_new = z_per_obj.max(dim=1)[0]
        z_new = torch.where(z_new > 0, z_new,
                            torch.tensor(0.01, device=grid_world.device, dtype=grid_world.dtype))
        grid_world[:, :, 2] = z_new
        return grid_world

    # --------------------------------------------------------------------- #
    # Tensors / motion library                                              #
    # --------------------------------------------------------------------- #

    def _setup_tensors(self) -> None:
        super()._setup_tensors()

        num_actors = self._root_states.shape[0] // self.num_envs  # humanoid + heightmaps + 3 objects
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        states_per_env = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])

        if not self.headless:
            num_objects = 3
            heightmap_idx = -self._num_heightmap_points - num_objects
            object_idx = -num_objects
            self._heightmap_states = states_per_env[..., heightmap_idx:object_idx, :]
            self._heightmap_pos = self._heightmap_states[..., :3]
            self._heightmap_rotation = self._heightmap_states[..., 3:7]
            heightmap_handles = to_torch(self._heightmap_handles, dtype=torch.int32, device=self.device)
            self._heightmap_actor_ids = (self._humanoid_actor_ids.unsqueeze(-1) + heightmap_handles).flatten()
        else:
            self._heightmap_pos = torch.zeros((self.num_envs, self._num_heightmap_points, 3),
                                              device=self.device)

        # The last three actors (box_large, box_small, chair).
        self._box_large_states = states_per_env[..., -3, :]
        self._box_small_states = states_per_env[..., -2, :]
        self._chair_states = states_per_env[..., -1, :]

        env_ids_int = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self._box_large_actor_ids = env_ids_int * num_actors + (num_actors - 3)
        self._box_small_actor_ids = env_ids_int * num_actors + (num_actors - 2)
        self._chair_actor_ids = env_ids_int * num_actors + (num_actors - 1)
        self._object_actor_ids = torch.cat(
            [self._box_large_actor_ids, self._box_small_actor_ids, self._chair_actor_ids], dim=0)

    def _refresh_sim_tensors(self) -> None:
        """Manually re-fetch the sim tensors. Heightmap pos must be preserved across refresh."""
        if not self.headless:
            heightmap_backup = self._heightmap_pos.clone()

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if not self.headless:
            self._heightmap_pos[:] = heightmap_backup

    def get_running_mean_size(self):
        return (self.get_obs_size(),)

    def _load_motion(self, motion_train_file, motion_test_file=None, subjects=None,
                     failed_keys=False) -> None:
        assert self._dof_offsets[-1] == self.num_dof
        if motion_test_file is None:
            motion_test_file = []
        if subjects is None:
            subjects = []

        if '_motion_lib' in self.__dict__:
            del self._motion_lib, self._motion_train_lib, self._motion_eval_lib
            torch.cuda.empty_cache()
            gc.collect()

        motion_lib_cfg = EasyDict({
            "motion_file": motion_train_file,
            "subjects": subjects,
            "failed_keys": failed_keys,
            "fix_height": FixHeightMode.ankle_fix,
            "min_length": self._min_motion_len,
            "max_length": self.max_len,
            "im_eval": flags.im_eval,
            "multi_thread": True,
            "smpl_type": self.humanoid_type,
            "randomrize_heading": True,
            "device": self.device,
            "load_imu": self.load_imu,
            "load_insole": self.load_insole,
            "load_slam": self.load_slam,
            "load_pose_p": self.load_pose_p,
            "load_joint_gp_l": self.load_joint_gp_l,
            "load_joint_gp_g": self.load_joint_gp_g,
            "load_jvel_gp": self.load_jvel_gp,
            "load_angv_gp": self.load_angv_gp,
            "load_rvel_gp": self.load_rvel_gp,
            "load_trans_gp": self.load_trans_gp,
            "load_imu_vel": self.load_imu_vel,
            "load_imu_angv": self.load_imu_angv,
            "load_joint_gt_g": self.load_joint_gt_g,
            "load_object": self.load_object,
            "win_len": self.win_len,
            "max_len": self.max_len,
            "rand_seq": self.rand_seq,
        })
        self._motion_lib = MotionLibSMPLGrip(motion_lib_cfg, gender_betas=self.humanoid_shapes.cpu())
        self._motion_lib.load_motions(
            skeleton_trees=self.skeleton_trees,
            gender_betas=self.humanoid_shapes.cpu(),
            limb_weights=self.humanoid_limb_and_weights.cpu(),
            random_sample=(not flags.test or self.rand_seq),
            max_len=-1 if flags.test else self.max_len,
        )
        self._motion_train_lib = self._motion_eval_lib = self._motion_lib
        self._set_object_pos()

    def _set_object_pos(self) -> None:
        if not self.load_object:
            return

        # Pull positions / rotations / vertices from the motion library.
        self.box_large_pos = self._motion_lib.box_large_pos
        self.box_large_rotation = self._motion_lib.box_large_rotation
        self.box_large_vertices = self._motion_lib.box_large_vertices
        self.box_small_pos = self._motion_lib.box_small_pos
        self.box_small_rotation = self._motion_lib.box_small_rotation
        self.box_small_vertices = self._motion_lib.box_small_vertices
        self.chair_pos = self._motion_lib.chair_pos
        self.chair_rotation = self._motion_lib.chair_rotation
        self.chair_vertices = self._motion_lib.chair_vertices

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        for state_buf, pos, rot in [
            (self._box_large_states, self.box_large_pos, self.box_large_rotation),
            (self._box_small_states, self.box_small_pos, self.box_small_rotation),
            (self._chair_states, self.chair_pos, self.chair_rotation),
        ]:
            state_buf[env_ids, 0:3] = pos[env_ids]
            state_buf[env_ids, 3:7] = rot[env_ids]

        # Two simulation steps to let Isaac Gym settle the static actors.
        for _ in range(2):
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(self._object_actor_ids),
                len(self._object_actor_ids),
            )
            self._refresh_sim_tensors()
            if _ == 0:
                self.gym.simulate(self.sim)
        torch.cuda.empty_cache()

    def forward_motion_samples(self) -> None:
        self.start_idx += self.num_envs
        self._motion_lib.load_motions(
            skeleton_trees=self.skeleton_trees,
            gender_betas=self.humanoid_shapes.cpu(),
            limb_weights=self.humanoid_limb_and_weights.cpu(),
            random_sample=self.rand_seq, start_idx=self.start_idx,
        )
        self._set_object_pos()
        self.reset()

    def resample_motions(self) -> None:
        if flags.test:
            self.forward_motion_samples()
            return
        self._motion_lib.load_motions(
            skeleton_trees=self.skeleton_trees,
            gender_betas=self.humanoid_shapes.cpu(),
            limb_weights=self.humanoid_limb_and_weights.cpu(),
            random_sample=(not flags.test or self.rand_seq) and (not self.seq_motions),
            max_len=-1 if flags.test else self.max_len,
        )
        time = (self.progress_buf * self.dt
                + self._motion_start_times + self._motion_start_times_offset)
        root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids, time)
        self._global_offset[:, :2] = self._humanoid_root_states[:, :2] - root_res['root_pos'][:, :2]
        self._set_object_pos()
        self.reset()

    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None,
                                        load_sensors=False):
        """Caches the most recent ``get_motion_state`` result keyed on (ids, times, offset)."""
        cache = self.ref_motion_cache
        if offset is None or "motion_ids" not in cache or cache['offset'] is None \
                or len(cache['motion_ids']) != len(motion_ids) \
                or len(cache['offset']) != len(offset) \
                or ((cache['motion_ids'] - motion_ids).abs().sum()
                    + (cache['motion_times'] - motion_times).abs().sum()
                    + (cache['offset'] - offset).abs().sum() > 0):
            cache['motion_ids'] = motion_ids.clone()
            cache['motion_times'] = motion_times.clone()
            cache['offset'] = offset.clone() if offset is not None else None
        else:
            return cache

        motion_res = self._motion_lib.get_motion_state(
            motion_ids, motion_times, offset=offset, load_sensors=load_sensors)
        del self.ref_motion_cache
        self.ref_motion_cache = motion_res
        return self.ref_motion_cache

    # --------------------------------------------------------------------- #
    # Observation construction                                              #
    # --------------------------------------------------------------------- #

    def get_task_obs_size(self) -> int:
        if not self._enable_task_obs:
            return 0
        track_size = len(self._track_bodies) * self._num_traj_samples * 24
        # IMU 6*7 + insole 2*5 + idf 6*18 + jdf 24*6 + heightmap 625
        return track_size + self.win_len * (6 * 7 + 2 * 5 + 6 * 18 + 24 * 6) + 625

    def get_task_obs_size_detail(self) -> OrderedDict:
        detail = OrderedDict()
        detail['target'] = self.get_task_obs_size()
        detail['fut_tracks'] = self._fut_tracks
        detail['num_traj_samples'] = self._num_traj_samples
        detail['obs_v'] = self.obs_v
        detail['track_bodies'] = self._track_bodies
        detail['models_path'] = self.models_path

        # Dev / progressive-net config.
        detail['num_prim'] = self.cfg['env'].get("num_prim", 2)
        detail['training_prim'] = self.cfg['env'].get("training_prim", 1)
        detail['actors_to_load'] = self.cfg['env'].get("actors_to_load", 2)
        detail['has_lateral'] = self.cfg['env'].get("has_lateral", True)

        track_body_size = self._num_joints * self._num_traj_samples * 24
        detail['target_size'] = (track_body_size
                                 + self.win_len * (6 * 7 + 2 * 5 + 6 * 18 + 24 * 6) + 625)
        return detail

    def _diff_observations(self, root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel,
                           ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel):
        """Build the IMU-difference and joint-difference observation tensors."""
        B, T, J, _ = ref_body_pos.shape
        _, _, I, _ = ref_body_rot.shape

        heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        hir_imu = heading_inv_rot.unsqueeze(-2).repeat((1, I, 1)).repeat_interleave(T, 0)
        hr_imu = heading_rot.unsqueeze(-2).repeat((1, I, 1)).repeat_interleave(T, 0)

        # Rotation differences (global → local around heading).
        diff_global_body_rot = torch_utils.quat_mul(
            ref_body_rot.reshape(B, T, I, 4),
            torch_utils.quat_conjugate(body_rot[:, None].repeat_interleave(T, 1)),
        )
        diff_local_body_rot_flat = torch_utils.quat_mul(
            torch_utils.quat_mul(hir_imu.reshape(-1, 4), diff_global_body_rot.reshape(-1, 4)),
            hr_imu.reshape(-1, 4),
        )
        diff_local_body_rot_flat = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)

        # Linear / angular velocity differences.
        diff_global_vel = ref_body_vel.reshape(B, T, I, 3) - body_vel.reshape(B, 1, I, 3)
        diff_local_vel = torch_utils.my_quat_rotate(hir_imu.reshape(-1, 4), diff_global_vel.reshape(-1, 3))
        diff_global_ang = ref_body_ang_vel.reshape(B, T, I, 3) - body_ang_vel.reshape(B, 1, I, 3)
        diff_local_ang = torch_utils.my_quat_rotate(hir_imu.reshape(-1, 4), diff_global_ang.reshape(-1, 3))

        # Reference body rotation in local (heading) frame.
        local_ref_body_rot = torch_utils.quat_mul(hir_imu.reshape(-1, 4), ref_body_rot.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

        # Reference body position relative to current root, in heading frame.
        diff_global_body_pos = ref_body_pos.reshape(B, T, J, 3) - body_pos.reshape(B, 1, J, 3)
        hir_jnt = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(T, 0)
        diff_local_body_pos_flat = torch_utils.my_quat_rotate(
            hir_jnt.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3))
        local_ref_body_pos = (ref_body_pos.reshape(B, T, J, 3) - root_pos.reshape(B, 1, 1, 3))
        local_ref_body_pos = torch_utils.my_quat_rotate(
            hir_jnt.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3))

        diff_local_body_rot_flat = diff_local_body_rot_flat.reshape(B, T, I, 6)
        diff_local_vel = diff_local_vel.reshape(B, T, I, 3)
        diff_local_ang = diff_local_ang.reshape(B, T, I, 3)
        local_ref_body_rot = local_ref_body_rot.reshape(B, T, I, 6)

        # Drop the IMU channels not exposed at this num_imus.
        keep = self.num_imus
        if keep < 6:
            diff_local_body_rot_flat[:, :, keep:, :] = 0.0
            diff_local_ang[:, :, keep:, :] = 0.0
            local_ref_body_rot[:, :, keep:, :] = 0.0

        idf_obs = torch.cat([diff_local_body_rot_flat, diff_local_vel,
                             diff_local_ang, local_ref_body_rot], dim=-1).reshape(B, T, -1)
        jdf_obs = torch.cat([
            diff_local_body_pos_flat.reshape(B, T, J, -1),
            local_ref_body_pos.reshape(B, T, J, -1),
        ], dim=-1).reshape(B, T, -1)
        return idf_obs, jdf_obs

    def _zero_unused_imu_channels(self, imu_tensor: torch.Tensor) -> torch.Tensor:
        """Zero IMU channels at indices ``[num_imus:]`` for tensors shaped [B, T, 6, *]."""
        if self.num_imus < 6:
            imu_tensor[:, :, self.num_imus:, :] = 0.0
        return imu_tensor

    def _compute_task_obs(self, env_ids=None, save_buffer: bool = True,
                          return_sensors: bool = True, reset: bool = False):
        """Build the full task observation for the given envs (or all envs)."""
        if env_ids is None:
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]

        # ---- Reference motion at this step (and possibly future window). ----
        if self._fut_tracks:
            time_steps = self._num_traj_samples
            B = env_ids.shape[0]
            time_int = (torch.arange(time_steps).to(self.device).repeat(B).view(-1, time_steps)
                        * self._traj_sample_timestep)
            motion_times = ((self.progress_buf[env_ids, None] + 1) * self.dt + time_int
                            + self._motion_start_times[env_ids, None]
                            + self._motion_start_times_offset[env_ids, None]).flatten()
            env_ids_steps = env_ids.repeat_interleave(time_steps)
            offset = self._global_offset[env_ids].repeat_interleave(time_steps, dim=0).view(-1, 3)
            motion_res = self._get_state_from_motionlib_cache(
                env_ids_steps, motion_times, offset, load_sensors=return_sensors)
        else:
            motion_times = ((self.progress_buf[env_ids] + 1) * self.dt
                            + self._motion_start_times[env_ids]
                            + self._motion_start_times_offset[env_ids])
            time_steps = 1
            motion_res = self._get_state_from_motionlib_cache(
                env_ids, motion_times, self._global_offset[env_ids], load_sensors=return_sensors)

        ref_root_pos, ref_root_rot, ref_dof_pos = motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"]
        ref_root_vel, ref_root_ang_vel, ref_dof_vel = motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"]
        ref_smpl_params, ref_limb_weights = motion_res["motion_bodies"], motion_res["motion_limb_weights"]
        ref_pose_aa, ref_rb_pos, ref_rb_rot = motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"]
        ref_body_vel, ref_body_ang_vel = motion_res["body_vel"], motion_res["body_ang_vel"]
        ref_mono_images = motion_res.get("mono_images")

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        # ---- Imitation observations (subset of tracked bodies). ----
        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]
        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
        ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
        ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
        ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]
        obs = humanoid_im.compute_imitation_observations_v6(
            root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset,
            ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset,
            time_steps, self._has_upright_start,
        )

        # ---- IMU observations (heading-frame-local). ----
        B, T, I, _ = motion_res["imu"].shape
        imu_acc = motion_res["imu"][..., :3].reshape(B, T, I, 3)
        imu_quat = motion_res["imu"][..., 3:].reshape(B, T, I, 4)
        imu_angv = motion_res["imu_angv"].reshape(B, T, I, 3)
        imu_vel = motion_res["imu_vel"].reshape(B, T, 6, 3)

        heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
        hir_imu = heading_inv_rot.unsqueeze(-2).repeat((1, I, 1)).repeat_interleave(T, 0)
        imu_quat_local = torch_utils.quat_mul(hir_imu.reshape(-1, 4), imu_quat.reshape(-1, 4))
        imu_acc_local = torch_utils.my_quat_rotate(hir_imu.reshape(-1, 4), imu_acc.reshape(-1, 3))
        imu_obs = torch.cat([imu_quat_local, imu_acc_local], dim=-1).reshape(B, T, I, 7)
        imu_obs = self._zero_unused_imu_channels(imu_obs)
        obs = torch.cat([obs, imu_obs.reshape(B, -1)], dim=-1)

        # ---- Insole, environment, joint-difference observations. ----
        insole = motion_res["insole"]            # (B, T, 2, 5)
        obs = torch.cat([obs, insole.reshape(B, -1)], dim=-1)
        obs = torch.cat([obs, self._heightmap_pos[env_ids][:, :, 2]], dim=-1)

        joint_p_g = motion_res["joint_p_g"]      # (B, T, 24, 3)
        joint_p_g = joint_p_g - joint_p_g[:, :, 0:1].repeat(1, 1, self._num_joints, 1)
        ref_pos = joint_p_g + (root_pos - joint_p_g[:, 0, 0])[:, None, None, :].repeat(
            1, T, self._num_joints, 1)
        ref_pos = ref_pos[:, :self.win_len]
        ref_rot = imu_quat[:, :self.win_len]
        ref_vel = imu_vel[:, :self.win_len]
        ref_ang = imu_angv[:, :self.win_len]

        body_rot_ee = body_rot[..., self.ee_rb_idx, :]
        body_vel_ee = body_vel[..., self.ee_rb_idx, :]
        body_ang_vel_ee = body_ang_vel[..., self.ee_rb_idx, :]

        idf_obs, jdf_obs = self._diff_observations(
            root_pos, root_rot, body_pos, body_rot_ee, body_vel_ee, body_ang_vel_ee,
            ref_pos, ref_rot, ref_vel, ref_ang,
        )
        obs = torch.cat([obs, torch.cat([idf_obs, jdf_obs], dim=-1).reshape(B, -1)], dim=-1)

        # ---- Bookkeeping: reference and current-estimate buffers. ----
        if save_buffer:
            self.ref_body_pos[env_ids] = ref_rb_pos
            self.ref_body_vel[env_ids] = ref_body_vel
            self.ref_body_rot[env_ids] = ref_rb_rot
            self.ref_body_ang_vel[env_ids] = ref_body_ang_vel
            self.ref_body_pos_subset[env_ids] = ref_rb_pos_subset
            self.ref_dof_pos[env_ids] = ref_dof_pos
            self.ref_dof_vel[env_ids] = ref_dof_vel
            self.ref_mono_images = ref_mono_images

        if not reset:
            self.current_body_rot[env_ids] = motion_res['pose_p'][:, 0, :, :]
            self.current_body_pos[env_ids] = motion_res['joint_p_g'][:, 0, :, :]
            self.current_root_vel[env_ids] = imu_vel[:, 0, 5, :]
            self.current_dof_pos[env_ids] = motion_res['dof_pos_p'][:, 0]
            self.current_dof_vel[env_ids] = motion_res['dof_vel_p'][:, 0]
            contact_left = insole[:, 0, 0, 3].bool() | insole[:, 0, 0, 4].bool()
            contact_right = insole[:, 0, 1, 3].bool() | insole[:, 0, 1, 4].bool()
            self.current_contact[env_ids] = torch.stack([contact_left, contact_right], dim=-1)

        del motion_res
        return obs

    # --------------------------------------------------------------------- #
    # Reset                                                                 #
    # --------------------------------------------------------------------- #

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        env_ids = self._normalize_env_ids(env_ids)
        if env_ids is None or env_ids.numel() == 0:
            return

        # Refresh the kinematics estimates so the freshly reset envs have valid current_*.
        self._compute_task_obs(env_ids, save_buffer=True, return_sensors=True, reset=True)

        # Reset history buffers to the current pose / velocity (no future drift assumed yet).
        root_pos = self._rigid_body_pos[env_ids, 0, :]
        root_vel = self.current_root_vel[env_ids]
        body_rot = self.current_body_rot[env_ids]
        body_pos = self.current_body_pos[env_ids]
        contact = self.current_contact[env_ids]
        if root_pos.dim() == 1:
            root_pos = root_pos.unsqueeze(0)
            root_vel = root_vel.unsqueeze(0)
            body_rot = body_rot.unsqueeze(0)
            body_pos = body_pos.unsqueeze(0)
            contact = contact.unsqueeze(0)

        self.root_pos_hist_buffer[env_ids] = root_pos.unsqueeze(1).repeat(1, _HIST_LEN, 1)
        self.root_vel_hist_buffer[env_ids] = root_vel.unsqueeze(1).repeat(1, _HIST_LEN, 1)
        self.body_rot_hist_buffer[env_ids] = body_rot.unsqueeze(1).repeat(1, _HIST_LEN, 1, 1)
        self.body_pos_hist_buffer[env_ids] = body_pos.unsqueeze(1).repeat(1, _HIST_LEN, 1, 1)
        self.contact_hist_buffer[env_ids] = contact.unsqueeze(1).repeat(1, _HIST_LEN, 1)

    def _normalize_env_ids(self, env_ids):
        """Coerce ``env_ids`` to a 1-D long tensor on ``self.device``; return ``None`` if empty."""
        if isinstance(env_ids, (list, tuple)):
            if len(env_ids) == 0:
                return None
            return torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return None
            return env_ids.unsqueeze(0) if env_ids.dim() == 0 else env_ids
        return torch.tensor([env_ids], dtype=torch.long, device=self.device)

    # --------------------------------------------------------------------- #
    # Fall recovery (test-time only, env.fall_recovery=True)                #
    # --------------------------------------------------------------------- #

    def _update_history_buffer(self) -> None:
        """Append the latest root / body / contact estimates to the rolling history."""
        idx = torch.arange(self.num_envs)
        root_pos = self._rigid_body_pos[:, 0, :]
        root_vel = self.current_root_vel[idx]
        body_rot = self.current_body_rot[idx]
        body_pos = self.current_body_pos[idx]
        contact = self.current_contact[idx]

        self.root_pos_hist_buffer = torch.cat(
            [self.root_pos_hist_buffer[:, 1:, :], root_pos.unsqueeze(1)], dim=1)
        self.root_vel_hist_buffer = torch.cat(
            [self.root_vel_hist_buffer[:, 1:, :], root_vel.unsqueeze(1)], dim=1)
        self.body_rot_hist_buffer = torch.cat(
            [self.body_rot_hist_buffer[:, 1:, :, :], body_rot.unsqueeze(1)], dim=1)
        self.body_pos_hist_buffer = torch.cat(
            [self.body_pos_hist_buffer[:, 1:, :, :], body_pos.unsqueeze(1)], dim=1)
        self.contact_hist_buffer = torch.cat(
            [self.contact_hist_buffer[:, 1:, :], contact.unsqueeze(1)], dim=1)

    def _check_and_handle_fall(self) -> None:
        """If the agent is falling, reset its state from a 1-second-old kinematic estimate."""
        # Only meaningful at test time, and only after the buffer has filled.
        if not flags.test:
            return
        current_step = self.progress_buf[0].item()
        if current_step < _HIST_LEN:
            return

        root_pos = self._rigid_body_pos[:, 0, :]
        is_fallen = (root_pos[:, 2] < _FALL_ROOT_Z) & (self.disc_probs < _FALL_DISC_PROB)
        fallen_env_ids = torch.nonzero(is_fallen, as_tuple=False).flatten()
        if len(fallen_env_ids) == 0:
            return

        old_pos = self.root_pos_hist_buffer[fallen_env_ids, 0]                     # (N, 3)

        # Initial (no-correction) pass to obtain a body-position trajectory we can use
        # to compute foot velocities for the contact-based correction.
        root_pos_hist = old_pos.unsqueeze(1) + torch.cumsum(
            self.dt * self.root_vel_hist_buffer[fallen_env_ids], dim=1)            # (N, 100, 3)
        body_rot_hist = self.body_rot_hist_buffer[fallen_env_ids]                  # (N, 100, 24, 4)
        body_pos_local = self.body_pos_hist_buffer[fallen_env_ids]                 # (N, 100, 24, 3)
        body_pos_hist = root_pos_hist.unsqueeze(2) + body_pos_local

        # Subtract foot drift while the foot is in contact (penetration / sliding fix).
        root_vel_hist = self.root_vel_hist_buffer[fallen_env_ids]
        foot_pos_hist = body_pos_hist[:, :, self.toe_rb_idx, :]
        foot_vel_hist = torch.zeros_like(foot_pos_hist)
        foot_vel_hist[:, :-1] = torch.diff(foot_pos_hist, dim=1) / self.dt
        foot_vel_hist[:, -1] = foot_vel_hist[:, -2]
        contact_hist = self.contact_hist_buffer[fallen_env_ids]                    # (N, 100, 2)
        contact_mask = contact_hist.unsqueeze(-1)
        num_contacts = contact_hist.sum(dim=-1, keepdim=True).clamp(min=1)
        contact_foot_vel = (foot_vel_hist * contact_mask).sum(dim=2) / num_contacts
        has_contact = contact_hist.any(dim=-1, keepdim=True)
        root_vel_hist = root_vel_hist - contact_foot_vel * has_contact
        root_pos_hist = old_pos.unsqueeze(1) + torch.cumsum(root_vel_hist * self.dt, dim=1)
        body_pos_hist = root_pos_hist.unsqueeze(2) + body_pos_local

        # Lift the body so that no joint penetrates the ground by more than a small margin.
        min_z = body_pos_hist[:, :, :, 2].min(dim=-1)[0]
        height_offset = torch.clamp(-min_z + 0.05, min=0.0)
        body_pos_hist[:, :, :, 2] += height_offset.unsqueeze(-1)
        root_pos_hist[:, :, 2] += height_offset
        assert body_pos_hist[:, :, :, 2].min() >= 0.0

        self.root_pos_hist_buffer[fallen_env_ids] = root_pos_hist
        self.root_vel_hist_buffer[fallen_env_ids] = root_vel_hist

        # Latest-frame state used to reset the simulator.
        reset_root_pos = root_pos_hist[:, -1, :]
        reset_root_vel = (reset_root_pos - root_pos_hist[:, -2, :]) / self.dt
        reset_root_rot = self.body_rot_hist_buffer[fallen_env_ids, -1, 0, :]
        prev_root_rot = self.body_rot_hist_buffer[fallen_env_ids, -2, 0, :]
        q_diff = torch_utils.quat_mul(reset_root_rot, torch_utils.quat_conjugate(prev_root_rot))
        reset_root_ang_vel = 2.0 * q_diff[:, :3] / self.dt

        self._humanoid_root_states[fallen_env_ids, 0:3] = reset_root_pos
        self._humanoid_root_states[fallen_env_ids, 3:7] = reset_root_rot
        self._humanoid_root_states[fallen_env_ids, 7:10] = reset_root_vel
        self._humanoid_root_states[fallen_env_ids, 10:13] = reset_root_ang_vel
        self._dof_pos[fallen_env_ids] = self.current_dof_pos[fallen_env_ids]
        self._dof_vel[fallen_env_ids] = self.current_dof_vel[fallen_env_ids]

        env_ids_int32 = fallen_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._humanoid_actor_ids[env_ids_int32]),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(self._humanoid_actor_ids[env_ids_int32]),
            len(env_ids_int32),
        )

        # Stash the corrected histories on ``extras`` so the player can rewrite
        # its own per-frame logs after each fall.
        valid_mask = body_pos_hist.abs().sum(dim=(1, 2, 3)) > 0.01
        valid_envs = torch.nonzero(valid_mask, as_tuple=True)[0]

        grf_hist = torch.zeros((len(fallen_env_ids), _HIST_LEN, 24, 3), device=self.device)
        grf_hist[:, :, self.toe_rb_idx[0], 2] = contact_hist[:, :, 0].float() * 9.81
        grf_hist[:, :, self.toe_rb_idx[1], 2] = contact_hist[:, :, 1].float() * 9.81

        if len(valid_envs) > 0:
            self.extras['fall_correction'] = {
                'env_ids': fallen_env_ids[valid_envs].cpu().numpy(),
                'buf_pos': body_pos_hist[valid_envs].cpu().numpy(),
                'buf_rot': body_rot_hist[valid_envs].cpu().numpy(),
                'buf_grf': grf_hist[valid_envs].cpu().numpy(),
                'current_step': current_step,
            }
        else:
            print("  Warning: insufficient buffer data, skipping history correction")

        print(f"Fall detected: resetting {len(fallen_env_ids)} humanoids")
        self.total_num_falls += len(fallen_env_ids)
        print(f"Total number of falls: {self.total_num_falls}")

    def _compute_fall_penalty(self) -> torch.Tensor:
        """Return per-env penalty when the agent has fallen (vs. ``ref_body_pos``)."""
        if not hasattr(self, 'ref_body_pos'):
            return torch.zeros(self.num_envs, device=self.device)
        height_diff = torch.abs(self._rigid_body_pos[..., 2] - self.ref_body_pos[..., 2])
        has_large_deviation = torch.any(height_diff > 0.3, dim=-1)
        return torch.where(
            has_large_deviation,
            -self.fall_penalty_scale,
            torch.zeros_like(has_large_deviation, dtype=torch.float),
        )

    # --------------------------------------------------------------------- #
    # Step / loop                                                           #
    # --------------------------------------------------------------------- #

    def _evaluate_discriminator(self) -> None:
        """Score the latest amp_obs against the AMP discriminator (uses cached normalizer)."""
        if self.disc_model is None:
            self.disc_probs = torch.ones(self.num_envs, device=self.device)
            return
        with torch.no_grad():
            amp_obs = self._amp_obs_buf.view(self.num_envs, -1)
            if self.amp_input_mean_std is not None:
                amp_obs = self.amp_input_mean_std(amp_obs)
            disc_logits = self.disc_model(amp_obs)
            self.disc_probs = torch.sigmoid(disc_logits).squeeze()

    def post_physics_step(self):
        super().post_physics_step()

        # The remaining bookkeeping is only useful at evaluation time.
        if not (flags.im_eval and flags.test):
            return

        self.extras['torque'] = self.dof_force_tensor.cpu().numpy()
        self.extras['grf_sim'] = self._contact_forces.cpu().numpy()

        self._evaluate_discriminator()
        self.extras['amp_obs'] = self._amp_obs_buf.view(self.num_envs, -1).cpu()
        self.extras['disc_probs'] = self.disc_probs.cpu()
        self.extras['fall_correction'] = None  # overwritten by _check_and_handle_fall on fall.

        # Optional kinematics-based fall recovery: opt-in via env.fall_recovery=True.
        if self.fall_recovery:
            self._update_history_buffer()
            self._check_and_handle_fall()

    def step(self, actions):
        self._update_heightmap()
        self.pre_physics_step(actions)
        self._physics_step()

        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        self.post_physics_step()
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

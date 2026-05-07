"""Visualize KinematicsNet inference output.

For each ``<seq>.npz`` produced by ``kinematics_net/inference.py``, render:

    * GT and predicted SMPL meshes (sharing the GT translation so the bodies
      overlap directly)
    * GT full / leaf joint spheres
    * Predicted joint spheres, root-aligned to the GT root each frame
    * A second predicted-joint trajectory reconstructed by integrating the
      predicted root velocity from frame 0
    * IMU rigid bodies + velocity arrows (GT and predicted)
    * A body-orbit tracking camera

The viewer assumes a z-up world frame.
"""
import argparse
import os
import random

import numpy as np
from scipy.signal import butter, filtfilt

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.scene.camera import PinholeCamera
from aitviewer.scene.node import Node
from aitviewer.viewer import Viewer


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# IMU placement on the body — must mirror data_process/kinematics_dataset.py.
IMU_KEYS = ['L_Foot', 'R_Foot', 'L_Wrist', 'R_Wrist', 'Head', 'Pelvis']
BODY_IDX = {key: idx for idx, key in enumerate(IMU_KEYS)}
IMU_VERT_IDX = [3438, 6838, 2208, 5669, 410, 3021]
# Joints to which IMUs are attached, used to recover GT IMU orientations from
# the SMPL rigged bodies.
IMU_JOINT_IDX = [10, 11, 20, 21, 15, 0]

# Subset of joints labelled as "leaf" — visualised separately as a sanity check.
LEAF_JOINT_IDX = [0, 10, 11, 15, 22, 23]

DT = 0.01  # 100 Hz inference

COLOR_GT = (149 / 255, 149 / 255, 149 / 255, 0.7)
COLOR_PRED = (51 / 255, 102 / 255, 153 / 255, 0.7)


# --------------------------------------------------------------------------- #
# Generic helpers                                                             #
# --------------------------------------------------------------------------- #

def butterworth_filter(data, cutoff=7.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def make_smpl_sequence(pose_body, pose_root, trans, smpl_layer, name, color):
    return SMPLSequence(
        poses_body=pose_body,
        poses_root=pose_root,
        betas=None,
        trans=trans,
        is_rigged=False,
        smpl_layer=smpl_layer,
        color=color,
        z_up=False,
        name=f'SMPL ({name})',
    )


def visualize_arrow(vectors, origins, name, color=(1, 0, 0, 1), magnitude=1.0):
    tips = origins + vectors * magnitude
    return Arrows(
        origins=origins.reshape(-1, 1, 3),
        tips=tips.reshape(-1, 1, 3),
        r_base=0.01, r_head=0.02, p=0.25,
        color=color, name=name,
    )


# --------------------------------------------------------------------------- #
# IMU renderables                                                             #
# --------------------------------------------------------------------------- #

def imu_rigid_bodies(imu_data, name, color=(1.0, 0.0, 1.0, 1.0)):
    rbs = Node(name=name)
    for key, payload in imu_data.items():
        rbs.add(RigidBodies(
            np.expand_dims(payload['pos'], axis=1),
            np.expand_dims(payload['ori'], axis=1),
            length=0.1, gui_affine=False, name=key, color=color, radius=0.02,
        ))
    return rbs


def imu_velocity_arrows(imu_data, name, color=(0, 0, 1, 1)):
    arrows = Node(name=name)
    for key, payload in imu_data.items():
        arrows.add(visualize_arrow(
            payload['vel'], payload['pos'], name=key, color=color, magnitude=0.5,
        ))
    return arrows


# --------------------------------------------------------------------------- #
# Tracking camera (z-up)                                                      #
# --------------------------------------------------------------------------- #

def body_tracking_camera(targets, viewer):
    targets = targets.copy()
    targets[:, 2] = 1.0
    targets = butterworth_filter(targets, cutoff=0.5)

    angles = np.linspace(np.radians(90), np.radians(450), num=2000)
    circle = np.column_stack((np.cos(angles) * 4,
                              np.sin(angles) * 4,
                              np.zeros_like(angles)))
    repeated = np.tile(circle, (targets.shape[0] // 1000 + 1, 1))[:targets.shape[0]]

    cam = PinholeCamera(
        targets + repeated, targets,
        viewer.window_size[0], viewer.window_size[1], viewer=viewer,
    )
    cam.name = 'Body Tracking Camera'
    return cam


# --------------------------------------------------------------------------- #
# Per-sequence rendering                                                      #
# --------------------------------------------------------------------------- #

def _imu_data_gt(vel_t, smpl_seq_gt):
    """IMU GT: position from SMPL vertices, velocity from inference output,
    orientation from the SMPL rigged bodies (so they exactly track the mesh)."""
    return {
        key: {
            'pos': smpl_seq_gt.vertices[:, IMU_VERT_IDX[BODY_IDX[key]]],
            'vel': vel_t[:, BODY_IDX[key]],
            'ori': smpl_seq_gt.rbs.rb_ori[:, IMU_JOINT_IDX[BODY_IDX[key]]],
        }
        for key in IMU_KEYS
    }


def _imu_data_pred(vel_p, RMB, smpl_seq_gt):
    """IMU prediction: positions snap to the GT mesh; velocity / orientation
    come from the model output."""
    return {
        key: {
            'pos': smpl_seq_gt.vertices[:, IMU_VERT_IDX[BODY_IDX[key]]],
            'vel': vel_p[:, BODY_IDX[key]],
            'ori': RMB[:, BODY_IDX[key]],
        }
        for key in IMU_KEYS
    }


def _trajectory_from_root_velocity(joints_raw, gt_root_t0, vel_p_root, dt=DT):
    """Frame-0-anchor the predicted joints to the GT root, then accumulate the
    predicted per-frame root velocity."""
    init_offset = gt_root_t0 - joints_raw[0, 0]
    cum_trans = np.cumsum(vel_p_root, axis=0) * dt
    return joints_raw + init_offset[None, None, :] + cum_trans[:, None, :]


def render_sequence(npz_data, smpl_layer, root_to_origin):
    """``root_to_origin`` is the constant translation that places the SMPL root
    at the world origin in t-pose."""
    pose_t = npz_data['pose_t'].reshape(-1, 24 * 3)
    tran_t = npz_data['tran_t'].reshape(-1, 3) + root_to_origin
    vel_t = npz_data['vel_t'].reshape(-1, 6, 3)

    pose_p = npz_data['pose_p'].reshape(-1, 24 * 3)
    vel_p = npz_data['vel_p'].reshape(-1, 6, 3)
    RMB = npz_data['RMB'].reshape(-1, 6, 3, 3)
    joints_p_raw = npz_data['joints_p'].reshape(-1, 24, 3)

    # GT and Pred SMPL meshes (both share GT translation so they overlay).
    smpl_seq_gt = make_smpl_sequence(
        pose_t[:, 3:], pose_t[:, :3], tran_t, smpl_layer, 'GT', COLOR_GT,
    )
    smpl_seq_pred = make_smpl_sequence(
        pose_p[:, 3:], pose_p[:, :3], tran_t, smpl_layer, 'Pred', COLOR_PRED,
    )

    full_joint_gt = smpl_seq_gt.joints.reshape(-1, 24, 3)
    leaf_joint_gt = full_joint_gt[:, LEAF_JOINT_IDX]
    full_joint_pred = smpl_seq_pred.joints.reshape(-1, 24, 3)

    # Per-frame root alignment of the predicted joints.
    joints_p = joints_p_raw + (full_joint_gt[:, 0:1] - joints_p_raw[:, 0:1])
    # Velocity-integrated trajectory (anchored to GT root at frame 0).
    joints_v = _trajectory_from_root_velocity(joints_p_raw, full_joint_gt[0, 0], vel_p[:, -1])

    mpjpe_smpl = np.mean(np.linalg.norm(full_joint_pred - full_joint_gt, axis=-1))
    mpjpe_joint = np.mean(np.linalg.norm(joints_p - full_joint_gt, axis=-1))
    print(f'    MPJPE-SMPL: {mpjpe_smpl:.4f}, MPJPE-JOINT: {mpjpe_joint:.4f}')

    # Joint spheres.
    leaf_gt_spheres = Spheres(positions=leaf_joint_gt, radius=0.03,
                              color=(1.0, 0.0, 0.0, 1.0), name='GT Leaf Joints')
    full_gt_spheres = Spheres(positions=full_joint_gt, radius=0.03,
                              color=(1.0, 0.0, 0.0, 1.0), name='GT Full Joints')
    pred_spheres = Spheres(positions=joints_p, radius=0.03,
                           color=(0.0, 0.0, 1.0, 1.0), name='Pred Joints')
    vel_spheres = Spheres(positions=joints_v, radius=0.03,
                          color=(0.0, 1.0, 0.0, 1.0), name='Vel-integrated Joints')

    # IMU dicts.
    imu_gt = _imu_data_gt(vel_t, smpl_seq_gt)
    imu_pred = _imu_data_pred(vel_p, RMB, smpl_seq_gt)

    # Compose viewer.
    viewer = Viewer()
    viewer.scene.add(smpl_seq_gt)
    viewer.scene.add(smpl_seq_pred)
    viewer.scene.add(leaf_gt_spheres)
    viewer.scene.add(full_gt_spheres)
    viewer.scene.add(pred_spheres)
    viewer.scene.add(vel_spheres)

    cam = body_tracking_camera(smpl_seq_gt.vertices[:, 0], viewer)
    viewer.scene.add(cam)
    viewer.set_temp_camera(cam)

    viewer.scene.add(imu_velocity_arrows(imu_gt, 'IMU Arrows (GT)', color=(1, 0, 0, 1)))
    viewer.scene.add(imu_velocity_arrows(imu_pred, 'IMU Arrows (Pred)', color=(0, 0, 1, 1)))
    viewer.scene.add(imu_rigid_bodies(imu_gt, 'IMU Rigid Bodies (GT)'))
    viewer.scene.add(imu_rigid_bodies(imu_pred, 'IMU Rigid Bodies (Pred)'))

    viewer.auto_set_floor = False
    viewer.scene.floor.enabled = True
    viewer.scene.origin.enabled = False
    viewer.scene.fps = 100.0
    viewer.playback_fps = 100.0
    viewer.shadows_enabled = True
    viewer.auto_set_camera_target = False
    viewer.run()
    viewer.close()


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Visualize KinematicsNet inference output.')
    parser.add_argument('--data_dir', type=str,
                        default='output/kinematics_net/infer',
                        help='Directory of per-sequence .npz files (matches kinematics_net/inference.py output).')
    parser.add_argument('--seq_filter', type=str, default='',
                        help='Only render sequences whose name contains this substring.')
    parser.add_argument('--shuffle', action='store_true',
                        help='Render in random order instead of sorted order.')
    args = parser.parse_args()

    C._conf.z_up = True
    smpl_layer = SMPLLayer(model_type='smpl', gender='neutral', device=C.device)
    # Constant translation that places the SMPL root at the world origin in t-pose.
    root_to_origin = -SMPLSequence.t_pose(smpl_layer=smpl_layer).joints[:, 0]

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith('.npz'))
    if args.seq_filter:
        files = [f for f in files if args.seq_filter in f]
    if not files:
        raise FileNotFoundError(
            f'No .npz files matched in {args.data_dir!r} with filter {args.seq_filter!r}.',
        )
    if args.shuffle:
        random.shuffle(files)

    print(f'Visualizing {len(files)} sequence(s) from {args.data_dir}')
    for fname in files:
        seq_name = os.path.splitext(fname)[0]
        print(f'>>> {seq_name}')
        render_sequence(
            dict(np.load(os.path.join(args.data_dir, fname))),
            smpl_layer, root_to_origin,
        )


if __name__ == '__main__':
    main()

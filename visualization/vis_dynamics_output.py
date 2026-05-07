"""Visualize DynamicsNet (GRIP) inference output.

For each ``<seq>.npz`` produced by ``dynamics_net/run_hydra.py`` (test loop),
render:

    * GT and predicted SMPL meshes (both anchored to the t-pose root)
    * The predicted mesh tinted by per-vertex torque magnitude (joint torque
      norms diffused onto vertices via a Gaussian weight map evaluated on the
      canonical pose, then mapped through the ``jet`` colormap)
    * A body-orbit tracking camera centred between the two figures
    * (PRISM only) per-take object meshes from the raw capture, if available

The viewer assumes a z-up world frame and the standard 24-joint SMPL skeleton.
"""
import argparse
import gc
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils.so3 import euler2aa_numpy, rot2aa_numpy
from aitviewer.viewer import Viewer


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

PRISM_RAW_DIR = '../MS-HPE/data/PRISM/integrated'

# Z offset that re-anchors the simulation root onto the floor.
ROOT_Z_OFFSET = 0.02

# Torque norm range used for the heatmap (clamped at this max).
TORQUE_CLAMP = 30.0

# First N frames are zeroed out before mapping torques to colors (warm-up spike).
TORQUE_WARMUP_FRAMES = 10

COLOR_GT = (149 / 255, 149 / 255, 149 / 255, 1.0)
COLOR_PRED = (51 / 255, 102 / 255, 153 / 255, 1.0)


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
        is_rigged=True,
        smpl_layer=smpl_layer,
        color=color,
        z_up=False,
        name=f'SMPL ({name})',
    )


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
# Torque heatmap                                                              #
# --------------------------------------------------------------------------- #

def joint_to_vertex_weights(smpl_layer, sigma=0.2):
    """Gaussian joint→vertex weight map evaluated on a canonical pose with the
    arms abducted 45° so the wrists are clear of the torso.

    Returns a tensor of shape ``[J, V]`` where each column is a softmax-like
    distribution over joints for that vertex.
    """
    pose_body = np.zeros((1, 23, 3))
    pose_body[:, 0] = euler2aa_numpy(np.array([0, 0, 45]), degrees=True)   # right arm out
    pose_body[:, 1] = euler2aa_numpy(np.array([0, 0, -45]), degrees=True)  # left arm out
    cpose = SMPLSequence(pose_body.reshape(1, -1), smpl_layer)

    vertices = torch.from_numpy(cpose.vertices[0])  # [V, 3]
    joints = torch.from_numpy(cpose.joints[0])      # [J, 3]
    distances = torch.norm(vertices.unsqueeze(0) - joints.unsqueeze(1), dim=2)  # [J, V]
    weights = torch.exp(-(distances ** 2) / (2 * sigma ** 2))
    return weights / (weights.sum(dim=0, keepdim=True) + 1e-8)


def torque_to_vertex_colors(torque_norm, weights, clamp=TORQUE_CLAMP):
    """Per-frame [F, J] joint-torque-norm tensor → per-frame [F, V, 4] RGBA via
    ``weights`` (joint→vertex diffusion) and the jet colormap."""
    joint_errors = torch.from_numpy(torque_norm).float()           # [F, J]
    vertex_errors = (joint_errors @ weights).clamp(0.0, clamp) / clamp  # [F, V]
    return plt.get_cmap('jet')(vertex_errors.cpu().numpy())        # [F, V, 4]


# --------------------------------------------------------------------------- #
# Object meshes (PRISM only)                                                  #
# --------------------------------------------------------------------------- #

def add_object_meshes(viewer, take_id):
    """Render PRISM scene objects for ``<subj>_<take>`` if the raw capture exists."""
    raw_path = os.path.join(PRISM_RAW_DIR, f'{take_id}.pkl')
    if not os.path.exists(raw_path):
        return
    with open(raw_path, 'rb') as f:
        raw = pickle.load(f)
    objects = raw.get('objects')
    if objects is None:
        return
    for obj_name, obj in objects.items():
        face_colors = np.ones((1, obj['faces'].shape[0], 4)) * np.array([0.5, 0.5, 0.5, 1.0])
        viewer.scene.add(Meshes(
            vertices=obj['vertices'], faces=obj['faces'],
            name=obj_name, face_colors=face_colors,
        ))


# --------------------------------------------------------------------------- #
# Per-sequence rendering                                                      #
# --------------------------------------------------------------------------- #

def _pad_first_frame(arr):
    """Repeat ``arr[0:1]`` once at the front. The dynamics_test loop drops the
    first warm-up frame; this restores it so timelines align with the GT."""
    return np.concatenate([arr[0:1], arr], axis=0)


def _split_pose(rot_mats):
    """[F, 24, 3, 3] rotation matrices → (pose_body [F, 23*3], pose_root [F, 3])."""
    aa = rot2aa_numpy(rot_mats.reshape(-1, 3, 3)).reshape(rot_mats.shape[0], -1, 3)
    return aa[:, 1:].reshape(-1, 23 * 3), aa[:, 0]


def render_sequence(npz_path, smpl_layer, heatmap_weights, with_objects):
    seq_name = os.path.splitext(os.path.basename(npz_path))[0]
    data = np.load(npz_path)

    pred_pos = _pad_first_frame(data['pred_pos'])
    pred_rot = _pad_first_frame(data['pred_rot'])
    gt_pos = _pad_first_frame(data['gt_pos'])
    gt_rot = _pad_first_frame(data['gt_rot'])
    torque = _pad_first_frame(data['torque'])

    pose_pred, ori_pred = _split_pose(pred_rot)
    pose_gt, ori_gt = _split_pose(gt_rot)
    trans_pred = pred_pos[:, 0].copy()
    trans_gt = gt_pos[:, 0].copy()
    trans_pred[:, 2] -= ROOT_Z_OFFSET
    trans_gt[:, 2] -= ROOT_Z_OFFSET

    # Re-anchor to t-pose root (so the SMPL viewer places the figure on the floor).
    tpose = SMPLSequence.t_pose(smpl_layer=smpl_layer)
    trans_offset = np.asarray(tpose.trans[0] - tpose.joints[0, 0])
    trans_pred += trans_offset
    trans_gt += trans_offset

    smpl_pred = make_smpl_sequence(pose_pred, ori_pred, trans_pred, smpl_layer,
                                   name='Pred', color=COLOR_PRED)
    smpl_gt = make_smpl_sequence(pose_gt, ori_gt, trans_gt, smpl_layer,
                                 name='GT', color=COLOR_GT)

    # Tint the predicted mesh by joint torque magnitude.
    torque_norm = np.linalg.norm(torque, axis=2)
    torque_norm[:TORQUE_WARMUP_FRAMES] = 0
    smpl_pred.mesh_seq.vertex_colors = torque_to_vertex_colors(torque_norm, heatmap_weights)

    # Tracking camera centred between the two roots.
    target = (smpl_pred.joints[:, 0] + smpl_gt.joints[:, 0]) / 2

    viewer = Viewer()
    viewer.scene.add(smpl_pred)
    viewer.scene.add(smpl_gt)
    cam = body_tracking_camera(target, viewer)
    viewer.scene.add(cam)
    viewer.set_temp_camera(cam)

    if with_objects:
        # take_id = first two underscore-separated tokens of the sequence name
        # (e.g. "subj001_take019" from "subj001_take019_seq001_000").
        take_id = '_'.join(seq_name.split('_')[:2])
        add_object_meshes(viewer, take_id)

    viewer.auto_set_floor = False
    viewer.scene.floor.enabled = True
    viewer.scene.origin.enabled = True
    viewer.scene.fps = 30.0
    viewer.playback_fps = 100.0
    viewer.shadows_enabled = True
    viewer.auto_set_camera_target = False
    viewer.run()
    viewer.close()
    gc.collect()


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Visualize DynamicsNet inference output.')
    parser.add_argument('--data_dir', type=str,
                        default='output/dynamics_net/results_success',
                        help='Directory of per-chunk .npz files (results_success or results_failed).')
    parser.add_argument('--seq_filter', type=str, default='',
                        help='Only render sequences whose name contains this substring.')
    parser.add_argument('--shuffle', action='store_true',
                        help='Render in random order instead of sorted order.')
    parser.add_argument('--with_objects', action='store_true',
                        help='Try to load PRISM scene objects from the raw capture (requires '
                             f'{PRISM_RAW_DIR}/<subj>_<take>.pkl).')
    args = parser.parse_args()

    C._conf.z_up = True
    smpl_layer = SMPLLayer(model_type='smpl', gender='neutral', device=C.device)
    heatmap_weights = joint_to_vertex_weights(smpl_layer)

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
            os.path.join(args.data_dir, fname),
            smpl_layer, heatmap_weights, args.with_objects,
        )


if __name__ == '__main__':
    main()

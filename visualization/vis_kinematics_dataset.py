"""Visualize the KinematicsNet dataset built by ``data_process/kinematics_dataset.py``.

Renders, per sequence:
    * the SMPL mesh from the saved ground-truth pose / translation
    * IMU rigid bodies + acceleration / velocity arrows
    * insole vertical-force arrows (per foot and combined)
    * foot contact points coloured by the contact label
    * a body-orbit and a foot-overlooking tracking camera

The dataset (and therefore this script) assumes a z-up world frame.
"""
import argparse
import os

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
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

# Foot contact landmark vertices on the SMPL mesh.
CONTACT_VERT_IDS = {'LF': 3222, 'LB': 3386, 'RF': 6620, 'RB': 6787}

# Foot-shape vertex anchors used to recover the world-frame insole CoP.
FOOT_VERT_IDS = {'L_Foot': [3220, 3386], 'R_Foot': [6622, 6787]}

# Visualization toggles per IMU.
VIS_IMU_RBS = {key: True for key in IMU_KEYS}
VIS_IMU_ARROWS = {key: True for key in IMU_KEYS}


# --------------------------------------------------------------------------- #
# Generic helpers                                                             #
# --------------------------------------------------------------------------- #

def butterworth_filter(data: np.ndarray, cutoff: float = 7.0,
                       fs: float = 100.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def make_smpl_sequence(pose_body, trans, pose_root, betas, smpl_layer,
                       color=(149 / 255, 149 / 255, 149 / 255, 1.0)) -> SMPLSequence:
    return SMPLSequence(
        poses_body=pose_body,
        poses_root=pose_root,
        betas=betas,
        trans=trans,
        is_rigged=False,
        smpl_layer=smpl_layer,
        color=color,
        z_up=True,
    )


def visualize_arrow(vectors: np.ndarray, origins: np.ndarray, name: str,
                    color=(1.0, 0.0, 0.0, 1.0), magnitude: float = 1.0) -> Arrows:
    tips = origins + vectors * magnitude
    return Arrows(
        origins=origins.reshape(-1, 1, 3),
        tips=tips.reshape(-1, 1, 3),
        r_base=0.01, r_head=0.02, p=0.25,
        color=color, name=name,
    )


# --------------------------------------------------------------------------- #
# Per-sequence renderables                                                    #
# --------------------------------------------------------------------------- #

def imu_rigid_bodies(sensor_data: dict, vis_flags: dict,
                     color=(1.0, 0.0, 1.0, 1.0)) -> Node:
    rbs = Node(name='IMU Rigid Bodies')
    for key, flag in vis_flags.items():
        if not flag or key not in sensor_data:
            continue
        payload = sensor_data[key]
        rbs.add(RigidBodies(
            np.expand_dims(payload['pos'], axis=1),
            np.expand_dims(payload['ori'], axis=1),
            length=0.2, gui_affine=False, name=key, color=color, radius=0.04,
        ))
    return rbs


def imu_acc_vel_arrows(sensor_data: dict, vis_flags: dict) -> Node:
    arrows_root = Node(name='Acc / Vel Arrows')
    for key, flag in vis_flags.items():
        if not flag or key not in sensor_data:
            continue
        per_imu = Node(name=key)
        per_imu.add(
            visualize_arrow(sensor_data[key]['acc'], sensor_data[key]['pos'],
                            name='acc', color=(1, 0, 0, 1), magnitude=0.05),
            visualize_arrow(sensor_data[key]['vel'], sensor_data[key]['pos'],
                            name='vel', color=(0, 0, 1, 1), magnitude=0.5),
        )
        arrows_root.add(per_imu)
    return arrows_root


def foot_contact_clouds(smpl_sequence: SMPLSequence, sensor_data: dict) -> Node:
    """Per-frame foot landmark spheres coloured red (off) / green (on)."""
    contact_node = Node(name='Foot Contact')
    side_specs = [
        ('L_Foot', [('LF', 0, 'Left Front'), ('LB', 1, 'Left Back')]),
        ('R_Foot', [('RF', 0, 'Right Front'), ('RB', 1, 'Right Back')]),
    ]
    for side, specs in side_specs:
        contacts = sensor_data[side]['contacts']  # [F, 2]
        F = contacts.shape[0]
        for vert_key, contact_col, display_name in specs:
            positions = smpl_sequence.vertices[:, CONTACT_VERT_IDS[vert_key]]
            colors = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (F, 1))
            colors[contacts[:, contact_col] == 1] = (0.0, 1.0, 0.0, 1.0)
            contact_node.add(PointClouds(
                positions, color=(1.0, 0.0, 0.0, 1.0),
                point_size=20.0, name=display_name, colors=colors,
            ))
    return contact_node


def world_vforce_arrows(insole_data: dict, smpl_verts: np.ndarray) -> dict:
    """Build per-foot and combined ground-reaction-force arrows in world frame."""
    cop_world_dict, vforce_vec_dict, origin_dict, vforce_arrows = {}, {}, {}, {}

    for side in ('L_Foot', 'R_Foot'):
        cop_local = insole_data[side]['CoP']
        foot_vert = smpl_verts[:, FOOT_VERT_IDS[side]]
        origin = (foot_vert[:, 0] + foot_vert[:, 1]) / 2
        length = np.linalg.norm(foot_vert[:, 0] - foot_vert[:, 1], axis=1)
        width = length / 2.78

        unit_x = (foot_vert[:, 0] - foot_vert[:, 1]) / length.reshape(-1, 1)
        unit_z = np.array([0, 0, 1]) if side == 'L_Foot' else np.array([0, 0, -1])
        unit_y = np.cross(unit_x, unit_z)

        cop_world = (
            origin
            + length.reshape(-1, 1) * cop_local[:, 0:1] * unit_x
            + width.reshape(-1, 1) * cop_local[:, 1:2] * unit_y
        )
        vforce = insole_data[side]['force']
        vforce_vec = np.zeros((vforce.shape[0], 3))
        vforce_vec[:, 2] = vforce.squeeze()

        cop_world_dict[side] = cop_world
        vforce_vec_dict[side] = vforce_vec
        origin_dict[side] = origin
        vforce_arrows[side] = visualize_arrow(
            vforce_vec.copy(), cop_world.copy(),
            name=f'vForce_{side}', color=(255 / 255, 215 / 255, 0 / 255, 1),
            magnitude=0.05,
        )

    F_L = insole_data['L_Foot']['force'].reshape(-1, 1)
    F_R = insole_data['R_Foot']['force'].reshape(-1, 1)
    F_total = F_L + F_R
    nonzero = (F_total > 1e-5).squeeze()
    F_total_safe = np.where(F_total == 0, 1e-8, F_total)
    cop_combined = (
        F_L * cop_world_dict['L_Foot'] + F_R * cop_world_dict['R_Foot']
    ) / F_total_safe
    cop_combined[~nonzero] = (
        (origin_dict['L_Foot'] + origin_dict['R_Foot']) / 2
    )[~nonzero]
    vforce_combined = vforce_vec_dict['L_Foot'] + vforce_vec_dict['R_Foot']
    vforce_arrows['combined'] = visualize_arrow(
        vforce_combined.copy(), cop_combined.copy(),
        name='vForce_combined', color=(255 / 255, 140 / 255, 0 / 255, 1),
        magnitude=0.05,
    )
    return vforce_arrows


# --------------------------------------------------------------------------- #
# Tracking cameras (z-up world frame)                                         #
# --------------------------------------------------------------------------- #

def _orbit_circle(num: int = 2000, radius: float = 4.0) -> np.ndarray:
    """Generate ``num`` (x, y, 0) points along a circle of given radius."""
    angles = np.linspace(np.radians(90), np.radians(450), num=num)
    return np.column_stack((np.cos(angles) * radius,
                            np.sin(angles) * radius,
                            np.zeros(angles.shape)))


def body_tracking_camera(smpl_sequence: SMPLSequence, viewer: Viewer) -> PinholeCamera:
    targets = smpl_sequence.vertices[:, 0].copy()
    targets[:, 2] = 1.0
    targets = butterworth_filter(targets, cutoff=0.5)

    circle = _orbit_circle()
    repeated = np.tile(circle, (targets.shape[0] // 1000 + 1, 1))[:targets.shape[0]]
    cam = PinholeCamera(
        targets + repeated, targets,
        viewer.window_size[0], viewer.window_size[1], viewer=viewer,
    )
    cam.name = 'Body Tracking Camera'
    return cam


def foot_tracking_camera(smpl_sequence: SMPLSequence, viewer: Viewer) -> PinholeCamera:
    targets = smpl_sequence.joints[:, 10].copy()  # left ankle
    targets = butterworth_filter(targets, cutoff=0.1)
    cam = PinholeCamera(
        targets + np.array([2.0, 2.0, 0.5]), targets,
        viewer.window_size[0], viewer.window_size[1], viewer=viewer,
    )
    cam.name = 'Foot Tracking Camera'
    return cam


# --------------------------------------------------------------------------- #
# Per-sequence rendering                                                      #
# --------------------------------------------------------------------------- #

def _build_sensor_data(seq: dict, smpl_sequence: SMPLSequence) -> dict:
    """Lift sensor-frame IMU readings into world frame and pair with body positions."""
    RIM, RIS, RSB = seq['RIM'], seq['RIS'], seq['RSB']
    aS, wS, vS = seq['aS'], seq['wS'], seq['vS']

    R = RIM.transpose(1, 2).matmul(RIS)
    RWB = R.matmul(RSB)
    aW = R.matmul(aS.unsqueeze(-1)).squeeze(-1)
    vW = R.matmul(vS.unsqueeze(-1)).squeeze(-1)
    # wW would follow the same pattern; not used by this viewer.

    sensor_data = {}
    for key in IMU_KEYS:
        idx = BODY_IDX[key]
        sensor_data[key] = {
            'acc': aW[:, idx].numpy(),
            'vel': vW[:, idx].numpy(),
            'pos': smpl_sequence.vertices[:, IMU_VERT_IDX[idx]],
            'ori': RWB[:, idx].numpy(),
        }

    insole = seq['insole']
    for idx, key in enumerate(['L_Foot', 'R_Foot']):
        sensor_data[key].update({
            'force': insole[:, idx, 0:1].numpy(),
            'CoP': insole[:, idx, 1:3].numpy(),
            'contacts': insole[:, idx, 3:5].numpy(),
        })
    return sensor_data


def render_sequence(seq: dict, smpl_layer: SMPLLayer, root_offset: np.ndarray) -> None:
    poses = seq['pose']
    pose_body = poses[:, 1:].reshape(-1, 69)
    pose_root = poses[:, 0]
    betas = np.zeros((poses.shape[0], 10))
    trans = seq['tran'] + root_offset

    smpl_sequence = make_smpl_sequence(pose_body, trans, pose_root, betas, smpl_layer)
    sensor_data = _build_sensor_data(seq, smpl_sequence)
    vforce_arrows = world_vforce_arrows(sensor_data, smpl_sequence.vertices)

    viewer = Viewer()
    viewer.scene.add(smpl_sequence)
    viewer.scene.add(imu_rigid_bodies(sensor_data, VIS_IMU_RBS))
    viewer.scene.add(imu_acc_vel_arrows(sensor_data, VIS_IMU_ARROWS))
    viewer.scene.add(vforce_arrows['L_Foot'])
    viewer.scene.add(vforce_arrows['R_Foot'])
    viewer.scene.add(vforce_arrows['combined'])
    viewer.scene.add(foot_contact_clouds(smpl_sequence, sensor_data))

    body_cam = body_tracking_camera(smpl_sequence, viewer)
    viewer.scene.add(body_cam)
    viewer.scene.add(foot_tracking_camera(smpl_sequence, viewer))
    viewer.set_temp_camera(body_cam)

    viewer.auto_set_floor = False
    viewer.scene.floor.enabled = True
    viewer.scene.origin.enabled = True
    viewer.scene.fps = 30.0
    viewer.playback_fps = 100.0
    viewer.shadows_enabled = True
    viewer.auto_set_camera_target = False
    viewer.run()
    viewer.close()


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize the KinematicsNet dataset.')
    parser.add_argument('--data_path', type=str,
                        default='data/preprocessed/kinematics_net/dataset_test.pt')
    parser.add_argument('--seq_filter', type=str, default='',
                        help='Only render sequences whose name contains this substring.')
    args = parser.parse_args()

    C._conf.z_up = True
    smpl_layer = SMPLLayer(model_type='smpl', gender='male', device=C.device)

    # Translation offset that places the SMPL root at the world origin in t-pose.
    root_offset = -SMPLSequence.t_pose(smpl_layer=smpl_layer).joints[:, 0]

    dataset = torch.load(args.data_path)
    indices = range(len(dataset['name']))
    if args.seq_filter:
        indices = [i for i in indices if args.seq_filter in dataset['name'][i]]
    if not indices:
        raise SystemExit(f'No sequences matched filter {args.seq_filter!r}.')

    print(f'Visualizing {len(indices)} sequence(s) from {args.data_path}')
    for seq_idx in indices:
        print(f'>>> {dataset["name"][seq_idx]}')
        seq = {k: dataset[k][seq_idx] for k in dataset}
        render_sequence(seq, smpl_layer, root_offset)


if __name__ == '__main__':
    main()

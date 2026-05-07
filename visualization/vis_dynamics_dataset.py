"""Visualize the DynamicsNet dataset built by ``data_process/dynamics_dataset.py``.

For each ``<seq_name>.pkl`` we render:

    * the SMPL mesh from the saved ground-truth pose / translation
    * the KinematicsNet predicted joints (``joints_p``)
    * the MuJoCo / SkeletonState joints (re-derived from ``pose_quat_global``)
    * IMU rigid bodies + velocity arrows (ground truth and prediction)
    * (PRISM only) per-take object meshes from the raw capture, if available
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch
import trimesh

# --------------------------------------------------------------------------- #
# Path setup                                                                  #
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "third_party"))

from aitviewer.configuration import CONFIG as C                            # noqa: E402
from aitviewer.models.smpl import SMPLLayer                                # noqa: E402
from aitviewer.renderables.arrows import Arrows                            # noqa: E402
from aitviewer.renderables.meshes import Meshes                            # noqa: E402
from aitviewer.renderables.rigid_bodies import RigidBodies                 # noqa: E402
from aitviewer.renderables.smpl import SMPLSequence                        # noqa: E402
from aitviewer.renderables.spheres import Spheres                          # noqa: E402
from aitviewer.scene.node import Node                                      # noqa: E402
from aitviewer.viewer import Viewer                                        # noqa: E402
from poselib.poselib.skeleton.skeleton3d import (                          # noqa: E402
    SkeletonState, SkeletonTree,
)


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

HUMANOID_MJCF_PATH = 'dynamics_net/data/assets/mjcf/smpl_humanoid.xml'
PRISM_RAW_DIR = '../MS-HPE/data/PRISM/integrated'

# IMU sensor placement (must mirror data_process/dynamics_dataset.py).
IMU_KEYS = ['L_Foot', 'R_Foot', 'L_Wrist', 'R_Wrist', 'Head', 'Pelvis']
BODY_IDX = {key: idx for idx, key in enumerate(IMU_KEYS)}
IMU_VERT_IDX = [3438, 6838, 2208, 5669, 410, 3021]


# --------------------------------------------------------------------------- #
# Reusable visualization helpers                                              #
# --------------------------------------------------------------------------- #

def set_smpl_sequence(pose_body, trans, ori, betas, gender, smpl_layer,
                      color=(149 / 255, 149 / 255, 149 / 255, 0.7)) -> SMPLSequence:
    return SMPLSequence(
        poses_body=pose_body,
        poses_root=ori,
        betas=betas,
        trans=trans,
        is_rigged=False,
        smpl_layer=smpl_layer,
        color=color,
        z_up=True,
    )


def visualize_arrow(vectors: np.ndarray, origins: np.ndarray, name: str,
                    color=(1, 0, 0, 1), magnitude: float = 1.0) -> Arrows:
    tips = origins + vectors * magnitude
    return Arrows(
        origins=origins.reshape(-1, 1, 3),
        tips=tips.reshape(-1, 1, 3),
        r_base=0.01, r_head=0.02, p=0.25,
        color=color, name=name,
    )


def vis_imu_rigid_bodies(imu_data: dict, name: str,
                         color=(1.0, 0.0, 1.0, 1.0)) -> Node:
    rbs = Node(name=name)
    for key, payload in imu_data.items():
        rb = RigidBodies(
            np.expand_dims(payload['pos'], axis=1),
            np.expand_dims(payload['ori'], axis=1),
            length=0.1, gui_affine=False, name=key, color=color, radius=0.02,
        )
        rbs.add(rb)
    return rbs


def vis_velocity_arrows(imu_data: dict, name: str, color=(0, 0, 1, 1)) -> Node:
    arrows = Node(name=name)
    for key, payload in imu_data.items():
        arrows.add(visualize_arrow(
            payload['vel'], payload['pos'], name=key, color=color, magnitude=0.5,
        ))
    return arrows


def vis_object_meshes(viewer: Viewer, take_id: str) -> None:
    """Render per-take PRISM object meshes if the raw capture is available."""
    raw_path = os.path.join(PRISM_RAW_DIR, f'{take_id}.pkl')
    if not os.path.exists(raw_path):
        return
    with open(raw_path, 'rb') as f:
        raw = pickle.load(f)
    raw_objects = raw.get('objects')
    if raw_objects is None:
        return
    for obj_name, obj in raw_objects.items():
        face_colors = np.ones((1, obj['faces'].shape[0], 4)) * np.array([0.5, 0.5, 0.5, 1.0])
        viewer.scene.add(Meshes(
            vertices=obj['vertices'], faces=obj['faces'],
            name=obj_name, face_colors=face_colors,
        ))


# --------------------------------------------------------------------------- #
# Per-pkl rendering                                                           #
# --------------------------------------------------------------------------- #

def _imu_dict(payload: dict, smpl_vertices: np.ndarray, key_suffix: str) -> dict:
    """Build an IMU dict suitable for ``vis_imu_rigid_bodies`` / ``vis_velocity_arrows``."""
    vW = payload['imu_data']['vel_gt'] if key_suffix == 'gt' else payload['imu_data']['vel_p']
    if isinstance(vW, torch.Tensor):
        vW = vW.numpy()
    # vel_p is saved flat as (F, 6*3); vel_gt is already (F, 6, 3).
    if vW.ndim == 2:
        vW = vW.reshape(vW.shape[0], len(IMU_KEYS), 3)
    if isinstance(payload['imu_data']['ori'], torch.Tensor):
        ori = payload['imu_data']['ori'].numpy()
    else:
        ori = payload['imu_data']['ori']

    return {
        key: {
            'pos': smpl_vertices[:, IMU_VERT_IDX[BODY_IDX[key]]],
            'vel': vW[:, BODY_IDX[key]],
            'ori': ori[:, BODY_IDX[key]],
        }
        for key in IMU_KEYS
    }


def render_sequence(pkl_path: str, smpl_layer: SMPLLayer,
                    skeleton_tree: SkeletonTree) -> None:
    seq_name = os.path.splitext(os.path.basename(pkl_path))[0]
    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)[seq_name]

    pose_aa = payload['pose_aa']
    pose_body = pose_aa[:, 3:]                             # [F, 69]
    pose_root = pose_aa[:, :3]                             # [F, 3]
    betas = payload['beta']                                # [F, 10]
    trans = payload['trans_orig']                          # [F, 3]

    smpl_sequence = set_smpl_sequence(
        pose_body, trans, pose_root, betas, payload['gender'], smpl_layer,
    )

    # KinematicsNet predicted joints.
    kn_joints = Spheres(
        payload['joints_p'], color=(255 / 255, 140 / 255, 0 / 255, 1.0),
        radius=0.02, name='Joints (KinematicsNet)',
    )

    # MuJoCo joints reconstructed from the saved global rotations.
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(payload['pose_quat_global']),
        payload['root_trans_offset'],
        is_local=False,
    )
    mujoco_joints = Spheres(
        sk_state.global_translation.numpy().astype(np.float32),
        radius=0.02, name='Joints (MuJoCo)', color=(0.0, 1.0, 0.0, 1.0),
    )

    # IMU rigid bodies + velocity arrows (GT and predicted).
    imu_gt = _imu_dict(payload, smpl_sequence.vertices, 'gt')
    imu_pred = _imu_dict(payload, smpl_sequence.vertices, 'pred')

    viewer = Viewer()
    viewer.scene.add(smpl_sequence)
    viewer.scene.add(kn_joints)
    viewer.scene.add(mujoco_joints)
    viewer.scene.add(vis_imu_rigid_bodies(imu_gt, 'IMU Rigid Bodies'))
    viewer.scene.add(vis_velocity_arrows(imu_gt, 'IMU Velocity (GT)', color=(0, 0, 1, 1)))
    viewer.scene.add(vis_velocity_arrows(imu_pred, 'IMU Velocity (Pred)', color=(0, 1, 0, 1)))

    if payload.get('objects') is not None:
        # The take id is the first 15 chars of the sequence name (e.g. ``subj001_take020``).
        vis_object_meshes(viewer, seq_name[:15])

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
    parser = argparse.ArgumentParser(description='Visualize the DynamicsNet dataset.')
    parser.add_argument('--data_dir', type=str,
                        default='data/preprocessed/dynamics_net/test',
                        help='Directory containing per-sequence .pkl files.')
    parser.add_argument('--seq_filter', type=str, default='',
                        help='Only render sequences whose name contains this substring.')
    args = parser.parse_args()

    C._conf.z_up = True
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)
    skeleton_tree = SkeletonTree.from_mjcf(HUMANOID_MJCF_PATH)

    pkl_paths = sorted(glob.glob(os.path.join(args.data_dir, '*.pkl')))
    if args.seq_filter:
        pkl_paths = [p for p in pkl_paths if args.seq_filter in os.path.basename(p)]
    if not pkl_paths:
        raise FileNotFoundError(
            f'No .pkl files matched in {args.data_dir!r} with filter {args.seq_filter!r}.',
        )

    print(f'Visualizing {len(pkl_paths)} sequence(s) from {args.data_dir}')
    for pkl_path in pkl_paths:
        seq_name = os.path.splitext(os.path.basename(pkl_path))[0]
        print(f'>>> {seq_name}')
        render_sequence(pkl_path, smpl_layer, skeleton_tree)


if __name__ == '__main__':
    main()

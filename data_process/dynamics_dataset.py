"""Build the DynamicsNet dataset.

For every sequence we combine:
    1. Ground-truth IMU + insole + SMPL data from the KinematicsNet dataset
    2. KinematicsNet inference output (pose_p, joints_p, vel_p)
    3. (PRISM only) per-take object metadata derived from the raw capture

and dump a single dictionary ``{seq_name: {...}}`` per sequence to
``data/preprocessed/dynamics_net/{train,test}/<seq_name>.pkl``.
"""
import argparse
import gc
import os
import pickle
import sys

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as sRot

# --------------------------------------------------------------------------- #
# Path setup so that vendored / sibling packages are importable               #
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "third_party"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "kinematics_net"))

import articulate as art                                          # noqa: E402
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree  # noqa: E402
from smpl_sim.smpllib.smpl_joint_names import (                   # noqa: E402
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot  # noqa: E402


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

SMPL_MODEL_PATH = 'data/smpl/SMPL_MALE.pkl'
PRISM_RAW_DIR = '../MS-HPE/data/PRISM/integrated'
PRISM_OBJECT_DIR = '../MS-HPE/data/PRISM/formatted/Object'
HUMANOID_MJCF_PATH = 'dynamics_net/data/assets/mjcf/smpl_humanoid.xml'

# IMU sensor placement (matches the KinematicsNet dataset's channel order).
IMU_KEYS = ['L_Foot', 'R_Foot', 'L_Wrist', 'R_Wrist', 'Head', 'Pelvis']

# Default extents (m) used as placeholders when an object is missing.
_OBJECT_DEFAULTS = {
    'BoxLarge': None, 'BoxSmall': None, 'Chair': None,
}


# --------------------------------------------------------------------------- #
# Geometric helpers                                                           #
# --------------------------------------------------------------------------- #

def reorder_box_vertices(corners: np.ndarray) -> np.ndarray:
    """Sort box-top corners by polar angle around their centroid."""
    xy = corners[:, :2]
    centroid = np.mean(xy, axis=0)
    angles = np.arctan2(xy[:, 1] - centroid[1], xy[:, 0] - centroid[0])
    return corners[np.argsort(angles)]


# --------------------------------------------------------------------------- #
# Humanoid model conversion (SMPL → MuJoCo)                                   #
# --------------------------------------------------------------------------- #

_ROBOT_CFG = {
    "mesh": False, "rel_joint_lm": True, "upright_start": True, "remove_toe": False,
    "real_weight": True, "real_weight_porpotion_capsules": True,
    "real_weight_porpotion_boxes": False, "replace_feet": True, "masterfoot": False,
    "big_ankle": True, "freeze_hand": True, "box_body": False, "master_range": 50,
    "body_params": {}, "joint_params": {}, "geom_params": {}, "actuator_params": {},
    "model": "smpl",
}

_SMPL_2_MUJOCO = [
    SMPL_BONE_ORDER_NAMES.index(q)
    for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES
]


def get_isaac_model(pose_aa: np.ndarray, root_trans: np.ndarray):
    """Convert axis-angle SMPL pose to the MuJoCo / SkeletonState representation."""
    # Ground-truth body shape is held neutral; the dataset is shape-agnostic.
    betas = np.zeros((1, 16))

    smpl_local_robot = LocalRobot(_ROBOT_CFG)
    smpl_local_robot.load_from_skeleton(
        betas=torch.from_numpy(betas[0:1]), gender=[0], objs_info=None,
    )
    smpl_local_robot.write_xml(HUMANOID_MJCF_PATH)
    skeleton_tree = SkeletonTree.from_mjcf(HUMANOID_MJCF_PATH)

    F = pose_aa.shape[0]
    pose_aa_mj = pose_aa.reshape(F, 24, 3)[:, _SMPL_2_MUJOCO]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(F, 24, 4)

    root_trans_t = torch.from_numpy(root_trans).float()
    root_trans_offset = root_trans_t + skeleton_tree.local_translation[0]

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, torch.from_numpy(pose_quat), root_trans_offset, is_local=True,
    )
    pose_quat_global = sk_state.global_rotation.numpy().astype(np.float32)
    pose_quat = sk_state.local_rotation.numpy()

    if _ROBOT_CFG["upright_start"]:
        # Re-align global rotations to MuJoCo's upright canonical pose.
        upright_inv = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        pose_quat_global = (
            sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy())
            * upright_inv
        ).as_quat().reshape(F, -1, 4)
        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset,
            is_local=False,
        )
        pose_quat = sk_state.local_rotation.numpy()

    return pose_quat_global, pose_quat, root_trans_offset


# --------------------------------------------------------------------------- #
# PRISM object metadata                                                       #
# --------------------------------------------------------------------------- #

# Mosh-frame → World-frame rotation (world is +z up).
_R_MOSH_TO_WORLD = sRot.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()


def _empty_object_metadata() -> dict:
    return {
        'center': np.array([0.0, 0.0, -10.0]),
        'extents': np.array([0.0, 0.0, -10.0]),
        'rotation': np.array([0.0, 0.0, 0.0, 1.0]),
        'vertices': np.array([[0.0, 0.0, -10.0]]).repeat(4, axis=0),
    }


def _stair_rotation_quat() -> np.ndarray:
    """Quaternion (x, y, z, w) used to align stair meshes with the world frame."""
    # scipy's quaternion convention is already (x, y, z, w), so no axis roll needed.
    return sRot.from_euler('xyz', [0, 0, 90], degrees=True).as_quat()


def load_prism_object_metadata(take_id: str, trial_name: str) -> dict:
    """Read the per-take object meshes and return placement metadata."""
    objects = {name: _empty_object_metadata() for name in _OBJECT_DEFAULTS}
    obj_dir_seq = os.path.join(PRISM_OBJECT_DIR, take_id)
    if not os.path.exists(obj_dir_seq):
        return objects

    for obj_file in sorted(os.listdir(obj_dir_seq)):
        obj_name = obj_file.split('.')[0]
        obj_data = trimesh.load(os.path.join(obj_dir_seq, obj_file))

        vertices = np.array(obj_data.vertices) @ _R_MOSH_TO_WORLD.T
        upper_verts = vertices[vertices[:, 2] > vertices[:, 2].mean()]
        upper_verts = reorder_box_vertices(upper_verts)
        assert upper_verts.shape[0] == 4

        objects[obj_name] = {
            'center': obj_data.bounding_box.centroid @ _R_MOSH_TO_WORLD.T,
            'extents': obj_data.bounding_box.extents @ _R_MOSH_TO_WORLD.T,
            'rotation': _stair_rotation_quat() if 'Stair' in trial_name
                        else np.array([0.0, 0.0, 0.0, 1.0]),
            'vertices': upper_verts,
        }
    return objects


# --------------------------------------------------------------------------- #
# Per-sequence build                                                          #
# --------------------------------------------------------------------------- #

def sensor_to_world(RIM, RIS, RSB, aS, wS, vS):
    """Transform IMU readings from sensor frame into world frame."""
    R = RIM.transpose(1, 2).matmul(RIS)
    return (
        R.matmul(RSB),                                          # RWB [F, 6, 3, 3]
        R.matmul(aS.unsqueeze(-1)).squeeze(-1),                 # aW  [F, 6, 3]
        R.matmul(wS.unsqueeze(-1)).squeeze(-1),                 # wW  [F, 6, 3]
        R.matmul(vS.unsqueeze(-1)).squeeze(-1),                 # vW  [F, 6, 3]
    )


def build_sequence(seq_name: str, kin_seq: dict, kinnet_output_dir: str,
                   body_model: art.ParametricModel, root_offset: np.ndarray,
                   dataset: str) -> dict:
    """Pack one sequence into the DynamicsNet pkl payload."""
    RIM = kin_seq['RIM']           # [6, 3, 3]
    RSB = kin_seq['RSB']           # [6, 3, 3]
    RIS = kin_seq['RIS']           # [F, 6, 3, 3]
    aS = kin_seq['aS']             # [F, 6, 3]
    wS = kin_seq['wS']             # [F, 6, 3]
    vS = kin_seq['vS']             # [F, 6, 3]
    tran = kin_seq['tran']         # [F, 3]
    poses = kin_seq['pose']        # [F, 24, 3]
    insole = kin_seq['insole']     # [F, 2, 5]

    # Sensor-frame → World-frame.
    RWB, aW, wW, vW = sensor_to_world(RIM, RIS, RSB, aS, wS, vS)

    # KinematicsNet inference output for this sequence.
    kn_data = dict(np.load(os.path.join(kinnet_output_dir, f'{seq_name}.npz')))
    pose_p = kn_data['pose_p']        # [F, 24, 3]
    joints_p = kn_data['joints_p']    # [F, 24, 3]
    vel_p = kn_data['vel_p']          # [F, 18]

    # SMPL forward kinematics → joints (root-anchored).
    pose_aa = np.concatenate([poses[:, 0].numpy(), poses[:, 1:].reshape(-1, 69).numpy()], axis=1)
    pose_rot = art.math.axis_angle_to_rotation_matrix(
        torch.from_numpy(pose_aa).float()
    ).reshape(-1, 24, 3, 3)
    tran_aligned = (tran + torch.from_numpy(root_offset).float()).numpy()
    _, joints_smpl, _ = body_model.forward_kinematics(
        pose=pose_rot, tran=torch.from_numpy(tran_aligned).float(), calc_mesh=False,
    )
    joints_smpl = joints_smpl.numpy()

    # MuJoCo / Skeleton state.
    pose_quat_global, pose_quat, root_trans_offset = get_isaac_model(pose_aa, tran_aligned)

    # PRISM-specific per-take object metadata.
    objects = None
    if dataset == 'PRISM':
        take_id = seq_name[:15]
        with open(os.path.join(PRISM_RAW_DIR, f'{take_id}.pkl'), 'rb') as f:
            raw = pickle.load(f)
        trial_name = raw["info"]['data_info']['trial_name']
        objects = load_prism_object_metadata(take_id, trial_name)

    return {
        'fps': 100,
        'pose_quat_global': pose_quat_global,
        'pose_quat': pose_quat,
        'root_trans_offset': root_trans_offset,
        'pose_aa': pose_aa,
        'trans_orig': tran_aligned,
        'pose_p': pose_p,
        'joints_p': joints_p,
        'joints_smpl': joints_smpl,
        'beta': np.zeros((poses.shape[0], 10)),
        'gender': 'male',
        'imu_data': {
            'ori': RWB,
            'acc': aW,
            'vel_gt': vW,
            'vel_p': vel_p,
        },
        'insole_data': insole,
        'objects': objects,
        'slam_traj': None,
    }


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

KINNET_INFER_DIR = 'output/kinematics_net/infer'


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the DynamicsNet dataset.')
    parser.add_argument('--dataset', type=str, default='PRISM', choices=['PRISM'])
    parser.add_argument('--n_imus', type=int, default=4, choices=[2, 3, 4, 5, 6],
                        help='Number of IMUs the KinematicsNet inference used.')
    parser.add_argument('--insole', type=lambda x: x.lower() == 'true', default=True,
                        help='Use the insole-aware KinematicsNet output.')
    args = parser.parse_args()

    kinnet_dataset_dir = 'data/preprocessed/kinematics_net'
    kinnet_output_dir = KINNET_INFER_DIR
    print(f'Dataset: {args.dataset}')
    print(f'KinematicsNet output directory: {kinnet_output_dir}')

    # Body model used to obtain root-anchored joint positions for each frame.
    body_model = art.ParametricModel(SMPL_MODEL_PATH)
    # T-pose root joint, used to shift the dataset translation onto the root.
    zero_pose = torch.eye(3).repeat(24, 1, 1).unsqueeze(0)
    _, t_pose_joints, _ = body_model.forward_kinematics(
        pose=zero_pose, tran=torch.zeros(1, 3), calc_mesh=False,
    )
    root_offset = -t_pose_joints[0, 0].numpy()

    seq_keys = ('RIM', 'RSB', 'RIS', 'aS', 'wS', 'mS', 'vS', 'tran', 'pose', 'insole')
    for mode in ('train', 'test'):
        data_path = os.path.join(kinnet_dataset_dir, f'dataset_{mode}.pt')
        dataset = torch.load(data_path)

        save_dir = f'data/preprocessed/dynamics_net/{mode}'
        os.makedirs(save_dir, exist_ok=True)

        n_seq = len(dataset['name'])
        for seq_idx in range(n_seq):
            seq_name = dataset['name'][seq_idx]
            print(f'>>> [{mode}] {seq_name}  ({seq_idx + 1}/{n_seq})')
            save_path = os.path.join(save_dir, f'{seq_name}.pkl')

            kin_seq = {k: dataset[k][seq_idx] for k in seq_keys}
            payload = build_sequence(
                seq_name, kin_seq, kinnet_output_dir,
                body_model, root_offset, args.dataset,
            )
            with open(save_path, 'wb') as f:
                pickle.dump({seq_name: payload}, f)

            del payload, kin_seq
            gc.collect()


if __name__ == '__main__':
    main()

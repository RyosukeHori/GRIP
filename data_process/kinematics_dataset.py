"""Dataset preprocessing for KinematicsNet.

Reads raw PRISM motion / IMU / insole captures and packs them into the tensor
dictionary consumed by ``kinematics_net/train.py`` and
``kinematics_net/inference.py``.
"""
import glob
import json
import os
import pickle
import sys

# Make ``kinematics_net/articulate`` importable when this script is run
# directly via ``python data_process/kinematics_dataset.py``.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'kinematics_net'))

import articulate as art  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


# --------------------------------------------------------------------------- #
# PRISM                                                                       #
# --------------------------------------------------------------------------- #

# IMU sensor placement on the body for the KinematicsNet pipeline.
IMU_KEYS = ['L_Foot', 'R_Foot', 'L_Wrist', 'R_Wrist', 'Head', 'Pelvis']

PRISM_RAW_DIR = 'data/PRISM'
SAVE_DIR = 'data/preprocessed/kinematics_net'
SMPL_MODEL_PATH = 'data/smpl/SMPL_MALE.pkl'
DATASET_SPLIT_PATH = 'data_process/json/dataset_split.json'

SEQ_LEN = 1000  # 10-second windows at 100 Hz


def _empty_dataset_dict() -> dict:
    return {'name': [], 'RIM': [], 'RSB': [], 'RIS': [], 'aS': [], 'wS': [], 'mS': [],
            'vS': [], 'tran': [], 'pose': [], 'insole': [], 'forces': []}


def _stack_imu_field(imu_dict: dict, field: str) -> torch.Tensor:
    """Stack one IMU field across IMU_KEYS into a [F, 6, ...] tensor."""
    return torch.from_numpy(
        np.stack([imu_dict[k][field] for k in IMU_KEYS], axis=1)
    ).float()


def _build_insole_tensor(insole_data: dict) -> tuple:
    """Pack the per-foot insole fields into ``[F, 2, 5]`` and ``[F, 2, 16]``."""
    foot_keys = ['L_Foot', 'R_Foot']
    force = torch.from_numpy(np.stack(
        [insole_data[k]['force'] for k in foot_keys], axis=1)).float()        # [F, 2, 1]
    cop = torch.from_numpy(np.stack(
        [insole_data[k]['CoP'] for k in foot_keys], axis=1)).float()          # [F, 2, 2]
    contact = torch.from_numpy(np.stack(
        [insole_data[k]['contacts'] for k in foot_keys], axis=1)).float()     # [F, 2, 2]
    insole = torch.cat([force, cop, contact], dim=-1)                         # [F, 2, 5]
    forces = torch.from_numpy(np.stack(
        [insole_data[k]['forces'] for k in foot_keys], axis=1)).float()       # [F, 2, 16]
    return insole, forces


def _drop_body_to_floor(tran: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    """Shift the root translation so the feet rest on z=0 + a small offset."""
    foot_height = np.zeros_like(tran)
    foot_height[:, 2] = (joints[0, 10, 2] + joints[0, 11, 2]) / 2
    return tran - foot_height + torch.tensor([0, 0, 0.02])


def _align_foot_imu_to_gt(acc_chunk: torch.Tensor, ori_chunk: torch.Tensor,
                          ori_gt_chunk: torch.Tensor) -> tuple:
    """Re-orient the foot IMUs so their initial frame matches the GT pose."""
    init_imu_ori_gt = ori_gt_chunk[0]
    foot_imu_acc = acc_chunk[:, 0:2, :].clone()
    foot_imu_ori = ori_chunk[:, 0:2, :].clone()
    for idx in range(2):
        init_inv = torch.inverse(foot_imu_ori[0, idx])
        foot_imu_ori[:, idx] = init_imu_ori_gt[idx] @ init_inv @ foot_imu_ori[:, idx]
        foot_imu_acc[:, idx] = foot_imu_acc[:, idx] @ init_inv.T @ init_imu_ori_gt[idx].T
    out_acc = acc_chunk.clone()
    out_ori = ori_chunk.clone()
    out_acc[:, 0:2, :] = foot_imu_acc
    out_ori[:, 0:2, :] = foot_imu_ori
    return out_acc, out_ori


def _world_to_sensor_frame(ori_chunk, acc_chunk, vel_gt_chunk):
    """Convert world-frame quantities into IMU sensor frame and synthesise wS/mS."""
    R_T = ori_chunk.transpose(2, 3)
    w_chunk = art.math.rotation_matrix_to_axis_angle(
        ori_chunk[:-1].transpose(2, 3).matmul(ori_chunk[1:])
    ).view(-1, ori_chunk.shape[1], 3) * 100
    w_chunk = torch.cat((w_chunk, torch.zeros_like(w_chunk[:1])))
    m_chunk = R_T.matmul(torch.tensor([1, 0, 0.]).unsqueeze(-1)).squeeze(-1)
    a_chunk = R_T.matmul(acc_chunk.unsqueeze(-1)).squeeze(-1)
    v_chunk = R_T.matmul(vel_gt_chunk.unsqueeze(-1)).squeeze(-1)
    return a_chunk, w_chunk, m_chunk, v_chunk


def process_prism() -> None:
    """Pack PRISM captures into the train/test tensor dictionaries."""
    body_model = art.ParametricModel(SMPL_MODEL_PATH)

    with open(DATASET_SPLIT_PATH) as f:
        split = json.load(f)
    train_seqs = set(split['train'])
    test_seqs = set(split['test'])

    pkl_paths = sorted(glob.glob(os.path.join(PRISM_RAW_DIR, 'subj*', 'take*.pkl')))
    if not pkl_paths:
        raise FileNotFoundError(
            f'No PRISM .pkl files found under {PRISM_RAW_DIR!r}. '
            'Unzip the PRISM dataset into that directory first.'
        )

    ret_train = _empty_dataset_dict()
    ret_test = _empty_dataset_dict()

    for pkl_path in pkl_paths:
        subj_id = os.path.basename(os.path.dirname(pkl_path))   # e.g. 'subj001'
        take_id = os.path.splitext(os.path.basename(pkl_path))[0]  # e.g. 'take002'

        print(f'Processing: {subj_id}/{take_id}.pkl')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        smpl_params = data['smpl_params']
        pose = torch.from_numpy(smpl_params['poses']).float().reshape(-1, 24, 3)
        tran = torch.from_numpy(smpl_params['trans']).float()

        # Forward kinematics → joints (used to drop the body to the floor).
        pose_rot = art.math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
        _, joints = body_model.forward_kinematics(pose=pose_rot, tran=tran, calc_mesh=False)
        tran = _drop_body_to_floor(tran, joints)

        # IMU + insole fields (new key names: ``acc_world_filt`` / ``ori_world`` / ``vel_world``).
        acc = _stack_imu_field(data['imu'], 'acc_world_filt')
        ori = _stack_imu_field(data['imu'], 'ori_world')
        ori_gt = _stack_imu_field(data['imu_gt'], 'ori_world')
        vel_gt = _stack_imu_field(data['imu_gt'], 'vel_world')
        insole, forces = _build_insole_tensor(data['insole'])

        # Slice into 10-second windows.
        for i in range(acc.shape[0] // SEQ_LEN):
            seq_name = f'{subj_id}_{take_id}_seq{i:03d}'
            sl = slice(i * SEQ_LEN, (i + 1) * SEQ_LEN)

            if seq_name in train_seqs:
                target = ret_train
            elif seq_name in test_seqs:
                target = ret_test
            else:
                continue  # sequence not assigned to any split

            acc_aligned, ori_aligned = _align_foot_imu_to_gt(acc[sl], ori[sl], ori_gt[sl])
            a_chunk, w_chunk, m_chunk, v_chunk = _world_to_sensor_frame(
                ori_aligned, acc_aligned, vel_gt[sl],
            )

            # Slice views share their parent's full storage; ``.clone()`` allocates
            # a per-sequence storage so ``torch.save`` writes only what the view
            # covers (otherwise sparse splits like ``test`` keep entire takes alive).
            target['name'].append(seq_name)
            target['RIM'].append(torch.eye(3).repeat(6, 1, 1))
            target['RSB'].append(torch.eye(3).repeat(6, 1, 1))
            target['tran'].append(tran[sl].clone())
            target['pose'].append(pose[sl].clone())
            target['insole'].append(insole[sl].clone())
            target['forces'].append(forces[sl].clone())
            target['RIS'].append(ori_aligned)            # already cloned in _align_foot_imu_to_gt
            target['aS'].append(a_chunk)                 # fresh tensor from matmul
            target['wS'].append(w_chunk)
            target['mS'].append(m_chunk)
            target['vS'].append(v_chunk)

    print(f"# Sequences: Train {len(ret_train['name'])}  Test {len(ret_test['name'])}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(ret_train, os.path.join(SAVE_DIR, 'dataset_train.pt'))
    torch.save(ret_test, os.path.join(SAVE_DIR, 'dataset_test.pt'))
    print(f'Saved to {SAVE_DIR}/dataset_{{train,test}}.pt')


if __name__ == '__main__':
    process_prism()

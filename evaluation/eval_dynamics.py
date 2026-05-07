"""Evaluate DynamicsNet (GRIP) inference output against the PRISM test set.

Pipeline:
  1. ``format_gt_data``       PRISM dynamics_net test pkl       → ``gt.pkl``
  2. ``format_grip_results``  output/dynamics_net/results_*     → ``grip.pkl``
  3. ``integrate_results``    align xy + butterworth grf        → ``integrated.pkl``
  4. ``evaluate``             metrics table + .xlsx

Run from the project root:

    python evaluation/eval_dynamics.py
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from aitviewer.configuration import CONFIG as C
C._conf.z_up = True
from aitviewer.models.smpl import SMPLLayer  # noqa: E402
from aitviewer.utils.so3 import rot2aa_numpy  # noqa: E402

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _PROJECT_ROOT)

from evaluation.pose_evaluator import PoseEvaluator  # noqa: E402

# Reference subject body weight used to denormalise GT GRF (insole reports
# force / body-weight). Matches the original eval_methods.py.
BODY_WEIGHT = 73.99543409049511

GRF_FILTER = dict(cutoff=7, fs=100, order=4)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def butterworth_filter(data, cutoff=7, fs=100, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def _smpl_joints(smpl_layer, ori, pose, trans):
    """Run SMPL forward kinematics with zero betas; return numpy (T, 24, 3)."""
    betas = np.zeros((pose.shape[0], 10), dtype=np.float32)
    _, joints = smpl_layer(
        poses_root=torch.from_numpy(ori).to(C.device).float(),
        poses_body=torch.from_numpy(pose).to(C.device).float(),
        betas=torch.from_numpy(betas).to(C.device).float(),
        trans=torch.from_numpy(trans).to(C.device).float(),
    )
    return joints[:, :24].cpu().numpy()


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# --------------------------------------------------------------------------- #
# Stage 1: GT formatting                                                      #
# --------------------------------------------------------------------------- #
def format_gt_data(test_dir, save_path, smpl_layer, seq_len=500):
    print('>>> Formatting GT Data')
    data_dict = {}

    for file in tqdm(sorted(os.listdir(test_dir))):
        if not file.endswith('.pkl'):
            continue
        stem = file.split('.')[0]
        seq = pickle.load(open(os.path.join(test_dir, file), 'rb'))[stem]

        ori = seq['pose_aa'][:, :3]
        pose = seq['pose_aa'][:, 3:]
        trans = seq['trans_orig']

        # insole_data: (T, 2, 5) — index 0 is BW-normalised GRF, indices 3:5
        # are toe/heel contact bools per foot.
        insole = _to_numpy(seq['insole_data'])
        contacts = insole[:, :, 3:]                  # (T, 2, 2)
        grf = insole[:, :, 0] * BODY_WEIGHT           # (T, 2) in Newtons

        joints = _smpl_joints(smpl_layer, ori, pose, trans)

        subj_id, take_id, seq_id = stem.split('_')
        n_chunks = pose.shape[0] // seq_len
        for c in range(n_chunks):
            s, e = c * seq_len, (c + 1) * seq_len
            data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[f'{c:03d}'] = {
                'pose': pose[s:e],
                'ori': ori[s:e],
                'trans': trans[s:e],
                'joints': joints[s:e],
                'contacts': contacts[s:e],
                'grf': grf[s:e],
            }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)


# --------------------------------------------------------------------------- #
# Stage 2: GRIP result formatting                                             #
# --------------------------------------------------------------------------- #
def format_grip_results(exp_dir, save_path, smpl_layer):
    print('>>> Formatting GRIP Results')
    data_dict = {}

    for mode in ('success', 'failed'):
        results_dir = os.path.join(exp_dir, f'results_{mode}')
        if not os.path.isdir(results_dir):
            continue
        for file in tqdm(os.listdir(results_dir), desc=mode):
            if not file.endswith('.npz'):
                continue
            data = dict(np.load(os.path.join(results_dir, file)))
            subj_id, take_id, seq_id, chunk_id = file.split('.')[0].split('_')

            # Pad first 2 frames so the chunk matches the GT seq_len of 500.
            # (DynamicsNet drops the first 2 frames as warm-up context.)
            pred_pos = np.concatenate((data['gt_pos'][0:2], data['pred_pos']), axis=0)
            pred_rot = np.concatenate((data['gt_rot'][0:2], data['pred_rot']), axis=0)
            contact_forces = np.concatenate((data['grf'][0:2], data['grf']), axis=0)  # (F, 24, 3)

            # Per-foot Z-axis force from contact joints.
            grf_left = contact_forces[:, 7, 2] + contact_forces[:, 10, 2]
            grf_right = contact_forces[:, 8, 2] + contact_forces[:, 11, 2]
            grf = np.stack((grf_left, grf_right), axis=1)  # (F, 2)

            F = pred_rot.shape[0]
            pred_aa = rot2aa_numpy(pred_rot.reshape(-1, 3, 3)).reshape(F, 24 * 3)
            ori = pred_aa[:, :3]
            pose = pred_aa[:, 3:]
            trans = pred_pos[:, 0]

            joints = _smpl_joints(smpl_layer, ori, pose, trans)

            data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[chunk_id] = {
                'pose': pose,
                'ori': ori,
                'trans': trans,
                'joints': joints,
                'grf': grf,
                'fail_flag': np.array([mode == 'failed'] * F),
            }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)


# --------------------------------------------------------------------------- #
# Stage 3: integrate GT + GRIP, align xy + filter GRF                         #
# --------------------------------------------------------------------------- #
def integrate_results(gt_path, grip_path, save_path):
    print('>>> Integrating Results')
    gt_data = pickle.load(open(gt_path, 'rb'))
    grip_data = pickle.load(open(grip_path, 'rb'))

    integrated = {}
    seq_ids = []

    for subj_id in sorted(gt_data):
        for take_id in sorted(gt_data[subj_id]):
            for seq_id in sorted(gt_data[subj_id][take_id]):
                for chunk_id in sorted(gt_data[subj_id][take_id][seq_id]):
                    gt = gt_data[subj_id][take_id][seq_id][chunk_id]
                    grip = grip_data.get(subj_id, {}).get(take_id, {}).get(seq_id, {}).get(chunk_id)
                    if grip is None:
                        continue

                    # Align GRIP xy to GT first frame; z is left untouched
                    # because GRIP already runs in the z-up world frame.
                    delta_xy = (gt['trans'][0] - grip['trans'][0])[:2]
                    grip_trans = grip['trans'].copy()
                    grip_joints = grip['joints'].copy()
                    grip_trans[:, :2] += delta_xy
                    grip_joints[:, :, :2] += delta_xy.reshape(1, 1, 2)

                    # Both GT and GRIP GRF go through the same butterworth
                    # so the comparison is fair (matches the reference impl).
                    gt_grf = butterworth_filter(gt['grf'], **GRF_FILTER)
                    grip_grf = butterworth_filter(grip['grf'], **GRF_FILTER)

                    integrated.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[chunk_id] = {
                        'GT': {**gt, 'grf': gt_grf},
                        'GRIP': {
                            'pose': grip['pose'],
                            'ori': grip['ori'],
                            'trans': grip_trans,
                            'joints': grip_joints,
                            'grf': grip_grf,
                        },
                        'fail_flag': grip['fail_flag'],
                    }
                    seq_ids.append(f'{subj_id}_{take_id}_{seq_id}_{chunk_id}')

    integrated['seq_ids'] = seq_ids
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(integrated, f)


# --------------------------------------------------------------------------- #
# Stage 4: metrics                                                            #
# --------------------------------------------------------------------------- #
_METRIC_KEYS = ['MPJPE-G', 'MPJPE-L', 'MPJPE-PA', 'MPJRE',
                'Acceleration Error', 'Foot Slide', 'FP', 'GRF Error']

_METRIC_HEADERS = {
    'MPJPE-G': 'MPJPE-G (mm)',
    'MPJPE-L': 'MPJPE-L (mm)',
    'MPJPE-PA': 'MPJPE-PA (mm)',
    'MPJRE': 'MPJRE (deg)',
    'Acceleration Error': 'Acc (m/s^2)',
    'Foot Slide': 'FS (m/s)',
    'FP': 'FP (mm)',
    'GRF Error': 'GRF (N)',
}

_COL_WIDTH = 14


def evaluate(integrated_path, excel_path, box=True):
    print('>>> Evaluating Results')
    data = pickle.load(open(integrated_path, 'rb'))
    seq_labels = data['seq_ids']

    evaluator = PoseEvaluator()
    accum = {key: [] for key in _METRIC_KEYS}

    for label in tqdm(seq_labels):
        subj_id, take_id, seq_id, chunk_id = label.split('_')
        chunk = data[subj_id][take_id][seq_id][chunk_id]
        gt = chunk['GT']
        pred = chunk['GRIP']

        poses_gt = np.concatenate((gt['ori'], gt['pose']), axis=1)
        poses_pred = np.concatenate((pred['ori'], pred['pose']), axis=1).astype(np.float32)

        joints_gt_global = gt['joints']
        joints_pred_global = pred['joints']
        joints_gt_local = joints_gt_global - joints_gt_global[:, 0:1]
        joints_pred_local = joints_pred_global - joints_pred_global[:, 0:1]

        metrics = evaluator.evaluate_sequence(
            torch.tensor(poses_pred).float(),
            torch.tensor(poses_gt).float(),
            torch.tensor(pred['trans']).float(),
            torch.tensor(gt['trans']).float(),
            torch.tensor(joints_pred_global).float(),
            torch.tensor(joints_gt_global).float(),
            torch.tensor(joints_pred_local).float(),
            torch.tensor(joints_gt_local).float(),
            torch.as_tensor(gt['contacts']),
            gt['grf'], pred['grf'], box,
        )
        for key in _METRIC_KEYS:
            accum[key].append(metrics[key])

    # Tabular print + xlsx export
    summary = {key: float(np.mean(accum[key])) for key in _METRIC_KEYS}

    n_cols = 1 + len(_METRIC_KEYS)
    table_width = n_cols * _COL_WIDTH + (n_cols - 1)

    print()
    print('=' * table_width)
    print('Evaluation Results')
    print('=' * table_width)
    headers = ['Method'] + [_METRIC_HEADERS[k] for k in _METRIC_KEYS]
    print(' '.join(f'{h:<{_COL_WIDTH}}' for h in headers))
    print('-' * table_width)
    row = ['GRIP'] + [f'{summary[k]:.2f}' for k in _METRIC_KEYS]
    print(' '.join(f'{v:<{_COL_WIDTH}}' for v in row))
    print('=' * table_width + '\n')

    df = pd.DataFrame([{'Method': 'GRIP', **{_METRIC_HEADERS[k]: summary[k] for k in _METRIC_KEYS}}])
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    df.to_excel(excel_path, index=False, sheet_name='Results')
    print(f'Results saved to: {excel_path}')


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='output/dynamics_net',
                        help='directory holding results_{success,failed}/ from dynamics_net/run_hydra.py')
    parser.add_argument('--gt-dir', default='data/preprocessed/dynamics_net/test',
                        help='directory of PRISM dynamics_net test sequences (matches dynamics_test.sh env.motion_file)')
    parser.add_argument('--out-dir', default='output/evaluation',
                        help='directory for the evaluation outputs (gt.pkl, grip.pkl, integrated.pkl, evaluation_results.xlsx)')
    parser.add_argument('--seq-len', type=int, default=500,
                        help='chunk length used by the inference run (must match dynamics_test.sh env.episode_length)')
    parser.add_argument('--no-box', action='store_true',
                        help='disable box/stair penetration buckets (PRISM scenes have boxes — leave on by default)')
    args = parser.parse_args()

    gt_path = os.path.join(args.out_dir, 'gt.pkl')
    grip_path = os.path.join(args.out_dir, 'grip.pkl')
    integrated_path = os.path.join(args.out_dir, 'integrated.pkl')
    excel_path = os.path.join(args.out_dir, 'evaluation_results.xlsx')

    smpl_layer = SMPLLayer(model_type='smpl', gender='male', device=C.device)

    format_gt_data(args.gt_dir, gt_path, smpl_layer, args.seq_len)
    format_grip_results(args.results_dir, grip_path, smpl_layer)
    integrate_results(gt_path, grip_path, integrated_path)
    evaluate(integrated_path, excel_path, box=not args.no_box)


if __name__ == '__main__':
    main()

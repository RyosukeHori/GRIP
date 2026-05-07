"""Run KinematicsNet inference over a dataset and dump per-sequence .npz files."""
import argparse
import os

import numpy as np
import torch
import tqdm

import articulate as art
from net import KinematicsNet
from _common import (
    J_IGNORE,
    J_REDUCE,
    build_input_features,
    build_label_targets,
    load_body_model,
    mask_unused_imus,
    select_device,
    transform_to_world,
)


# --------------------------------------------------------------------------- #
# Per-sequence inference                                                      #
# --------------------------------------------------------------------------- #

def _decode_pose(RRJ_pred: torch.Tensor, body_model: art.ParametricModel,
                 device: torch.device) -> torch.Tensor:
    """Decode the network's 6D rotation prediction into [F, 24, 3] axis-angle pose."""
    F = RRJ_pred.shape[0]
    pose_6d = art.math.r6d_to_rotation_matrix(RRJ_pred).reshape(-1, 16, 3, 3)

    # Fill the kept-joint global rotations, then run inverse kinematics for the rest.
    glb_pose = torch.eye(3).repeat(pose_6d.shape[0], 24, 1, 1).to(device)
    glb_pose[:, J_REDUCE[1:]] = pose_6d[:, 1:]

    pose_p = body_model.inverse_kinematics_R(glb_pose).view(F, 24, 3, 3)
    pose_p[:, 0] = pose_6d[:, 0].reshape(F, 3, 3)
    pose_p[:, J_IGNORE] = torch.eye(3).to(device)

    return art.math.rotation_matrix_to_axis_angle(pose_p).reshape(F, 24, 3)


def inference_sequence(model: KinematicsNet, seq: dict, body_model: art.ParametricModel,
                       device: torch.device, n_imus: int, insole: bool) -> dict:
    """Run KinematicsNet on a single sequence and return GT / predicted tensors."""
    pose = seq['pose']
    tran = seq['trans']

    # 1) Sensor → world frame.
    RMB, aM, wM, vM = transform_to_world(
        seq['aS'], seq['wS'], seq['vS'], seq['RIS'], seq['RIM'], seq['RSB'],
    )

    # 2) Optionally drop high-index IMUs.
    RMB, aM, wM = mask_unused_imus(RMB, aM, wM, n_imus)

    # 3) Build feature and label tensors.
    input_data = build_input_features(aM, wM, RMB, seq['insole'], insole=insole)
    label_data, _ = build_label_targets(pose, vM, body_model, device)

    # 4) Forward pass on the sequence as a batch of 1.
    model.eval()
    x = input_data.to(device).unsqueeze(0)              # [1, F, 100]
    y = label_data.to(device).unsqueeze(0)              # [1, F, 246]
    with torch.no_grad():
        output = model(x, y[:, 0])
    output, x, y = output.squeeze(0), x.squeeze(0), y.squeeze(0)

    # 5) Slice predictions and ground truth out of the flat tensors.
    F = pose.shape[0]
    pRL_pred, pRJ_pred = output[:, :15],   output[:, 15:84]
    vM_pred,  RRJ_pred = output[:, 84:102], output[:, 102:198]
    pRL_gt,   pRJ_gt   = y[:, :15],         y[:, 15:84]

    # Reshape into convenient layouts and reattach a zero-root joint for the
    # ``joints`` arrays so they map cleanly back to the SMPL skeleton.
    zero_root = torch.zeros(F, 1, 3)
    return {
        'pose_t': pose.cpu().numpy(),
        'tran_t': tran.cpu().numpy(),
        'l_joints_t': pRL_gt.reshape(-1, 5, 3).cpu().numpy(),
        'joints_t': torch.cat([zero_root, pRJ_gt.cpu().reshape(-1, 23, 3)], dim=1),
        'vel_t': vM.flatten(1).cpu().numpy(),

        'pose_p': _decode_pose(RRJ_pred, body_model, device).cpu().numpy(),
        'l_joints_p': pRL_pred.reshape(-1, 5, 3).cpu().numpy(),
        'joints_p': torch.cat([zero_root, pRJ_pred.cpu().reshape(-1, 23, 3)], dim=1),
        'vel_p': vM_pred.cpu().numpy(),

        'RMB': RMB.to(device).cpu().numpy(),
    }


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def run_inference(data_path: str, output_dir: str, n_imus: int, insole: bool) -> None:
    device = select_device()
    print(f"Using device: {device}")

    body_model = load_body_model(device)

    model_path = os.path.join(output_dir, 'models', 'best_model.pt')
    print(f"Loading model: {model_path}")
    model = KinematicsNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    infer_dir = os.path.join(output_dir, 'infer')
    os.makedirs(infer_dir, exist_ok=True)

    print(f"Loading data: {data_path}")
    data = torch.load(data_path, map_location='cpu', weights_only=True)
    num_sequences = len(data['pose'])
    print(f"Number of sequences: {num_sequences}")

    seq_keys = ('pose', 'aS', 'wS', 'vS', 'RIS', 'RIM', 'RSB', 'insole')
    for seq_idx in tqdm.tqdm(range(num_sequences), desc='Processing sequences'):
        seq = {k: data[k][seq_idx] for k in seq_keys}
        seq['trans'] = data['tran'][seq_idx]

        result = inference_sequence(model, seq, body_model, device,
                                    n_imus=n_imus, insole=insole)

        seq_name = data['name'][seq_idx]
        np.savez(os.path.join(infer_dir, f"{seq_name}.npz"), **result)

    print(f"Inference complete! Results saved: {infer_dir}")


OUTPUT_DIR = 'output/kinematics_net'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRIP KinematicsNet Inference')
    parser.add_argument('--dataset', type=str, default='PRISM', choices=['PRISM'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--n_imus', type=int, default=4, choices=[2, 3, 4, 5, 6],
                        help='number of IMUs')
    parser.add_argument('--insole', type=lambda x: x.lower() == 'true', default=True,
                        help='use insole data')
    parser.add_argument('--fps', type=int, default=100, help='Frame rate')
    args = parser.parse_args()

    data_path = f'data/preprocessed/kinematics_net/dataset_{args.mode}.pt'
    print(f"Output directory: {OUTPUT_DIR}")

    run_inference(data_path, OUTPUT_DIR, args.n_imus, args.insole)

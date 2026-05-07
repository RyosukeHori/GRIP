"""Shared helpers used by kinematics_net training and inference.

Centralises the body-model loading, IMU masking, and sensor-to-world transforms
that would otherwise be duplicated across train.py and inference.py.
"""
from __future__ import annotations

import torch
import articulate as art


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Vertex indices on the SMPL mesh that approximate IMU sensor positions.
V_IMU = (1961, 5424, 1176, 4662, 411, 3021)

# Joint reduction map (16 kept, 8 ignored).
J_REDUCE = (0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
J_IGNORE = (7, 8, 10, 11, 20, 21, 22, 23)
J_CONTACT = (0, 10, 11, 22, 23)

SMPL_MODEL_PATH = 'data/smpl/SMPL_MALE.pkl'

NUM_IMUS = 6


# --------------------------------------------------------------------------- #
# Device / model helpers                                                      #
# --------------------------------------------------------------------------- #

def select_device(cuda_device: int = 0) -> torch.device:
    """Return CUDA / MPS / CPU device, defaulting to a specific CUDA index."""
    if torch.cuda.is_available():
        if cuda_device >= torch.cuda.device_count():
            cuda_device = 0
        return torch.device(f'cuda:{cuda_device}')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_body_model(device: torch.device) -> art.ParametricModel:
    """Load the IMU-masked SMPL parametric model used for label generation."""
    return art.ParametricModel(SMPL_MODEL_PATH, vert_mask=V_IMU, device=device)


# --------------------------------------------------------------------------- #
# IMU + insole feature pipeline                                               #
# --------------------------------------------------------------------------- #

def mask_unused_imus(RMB: torch.Tensor, aM: torch.Tensor, wM: torch.Tensor,
                     n_imus: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-out IMU channels above ``n_imus`` and reset their orientations to identity.

    Operates in place to mirror the original behaviour. ``n_imus`` is clamped so
    callers passing the full count get a no-op.
    """
    if n_imus >= NUM_IMUS or n_imus < 2:
        return RMB, aM, wM
    n_zero = NUM_IMUS - n_imus
    RMB[:, n_imus:] = torch.eye(3).repeat(n_zero, 1, 1)
    aM[:, n_imus:] = torch.zeros(n_zero, 3)
    wM[:, n_imus:] = torch.zeros(n_zero, 3)
    return RMB, aM, wM


def transform_to_world(aS: torch.Tensor, wS: torch.Tensor, vS: torch.Tensor,
                       RIS: torch.Tensor, RIM: torch.Tensor, RSB: torch.Tensor):
    """Transform IMU acceleration / angular velocity / linear velocity into world frame.

    Returns ``(RMB, aM, wM, vM)`` where each tensor has shape ``[F, 6, ...]``.
    """
    R = RIM.transpose(1, 2).matmul(RIS)
    RMB = R.matmul(RSB)                                            # [F, 6, 3, 3]
    aM = R.matmul(aS.unsqueeze(-1)).squeeze(-1)                    # [F, 6, 3]
    wM = R.matmul(wS.unsqueeze(-1)).squeeze(-1)                    # [F, 6, 3]
    vM = R.matmul(vS.unsqueeze(-1)).squeeze(-1)                    # [F, 6, 3]
    return RMB, aM, wM, vM


def build_input_features(aM: torch.Tensor, wM: torch.Tensor, RMB: torch.Tensor,
                         iS: torch.Tensor, insole: bool = True) -> torch.Tensor:
    """Stack IMU + insole signals into the [F, 100] feature tensor consumed by the network."""
    if not insole:
        iS = torch.zeros_like(iS)
    return torch.cat([
        aM.flatten(1),    # [F, 18]   acceleration
        wM.flatten(1),    # [F, 18]   angular velocity
        RMB.flatten(1),   # [F, 54]   orientation (6 IMUs x 3x3)
        iS.flatten(1),    # [F, 10]   insole forces / contact
    ], dim=1)


def build_label_targets(pose: torch.Tensor, vM: torch.Tensor,
                        body_model: art.ParametricModel, device: torch.device):
    """Compute supervision targets from SMPL pose and IMU velocities.

    Returns ``(targets, pose_mat)`` where ``targets`` is the [F, 246] tensor
    ``[pRL(15) | pRJ(69) | vM(18) | RRJ(144)]`` and ``pose_mat`` is the per-frame
    rotation-matrix pose that produced it.
    """
    pose_mat = art.math.axis_angle_to_rotation_matrix(pose.view(-1, 3)).view(-1, 24, 3, 3).to(device)
    _, j_global, v_global = body_model.forward_kinematics(pose_mat, calc_mesh=True)

    pRL = (v_global[:, :5] - v_global[:, 5:]).flatten(1)            # [F, 15]
    pRJ = j_global[:, 1:].flatten(1)                                # [F, 69]
    vM_flat = vM.flatten(1).to(device)                              # [F, 18]
    RRJ = pose_mat[:, list(J_REDUCE)].flatten(1).to(device)         # [F, 144]

    targets = torch.cat([pRL, pRJ, vM_flat, RRJ], dim=1)            # [F, 246]
    return targets, pose_mat

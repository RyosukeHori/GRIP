"""Pose evaluation metrics used by ``eval_dynamics.py``.

The metric implementations here mirror ``eval_pose.py`` from the research
codebase: kinematics (MPJPE-G/L/PA, MPJRE, RTE, Acceleration, Jitter), foot
behaviour (slide, penetration / floating, height-at-contact, velocity) and
GRF error.
"""
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'kinematics_net'))

import articulate as art  # noqa: E402

# SMPL parent indices for the 24-joint kinematic tree.
_SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


class PoseEvaluator:
    def __init__(self):
        self._m2mm = 1000

    # ------------------------------------------------------------------ #
    # Stability                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_acceleration(joints, frame_rate=100):
        """Joint acceleration (m/s^2) from per-frame positions (F, 24, 3)."""
        dt = 1 / frame_rate
        velocity = (joints[1:] - joints[:-1]) / dt
        return (velocity[1:] - velocity[:-1]) / dt

    @staticmethod
    def _compute_jitter(joints):
        """Average jerk magnitude (m/s^3) over the sequence."""
        jerk = torch.diff(joints, n=3, dim=0)
        return torch.mean(torch.norm(jerk, dim=-1))

    # ------------------------------------------------------------------ #
    # Pose error                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _global_rotation(pose_aa):
        """Forward-kinematics on rotations only — returns (F, 24, 3, 3)."""
        F = pose_aa.shape[0]
        local = art.math.axis_angle_to_rotation_matrix(pose_aa).view(F, 24, 3, 3)
        global_rot = torch.eye(3, device=pose_aa.device).expand(F, 24, 3, 3).clone()
        for i, parent in enumerate(_SMPL_PARENTS):
            if parent == -1:
                global_rot[:, i] = local[:, i]
            else:
                global_rot[:, i] = torch.matmul(global_rot[:, parent], local[:, i])
        return global_rot

    @staticmethod
    def _rotation_angle(R_diff):
        """Geodesic angle (radians) of a batch of 3x3 rotation matrices."""
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        return torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))

    def compute_mpjre(self, pred_pose, gt_pose):
        """Mean Per Joint Rotation Error (degrees) over the 24-joint tree."""
        pred_global = self._global_rotation(pred_pose)
        gt_global = self._global_rotation(gt_pose)

        total = 0.0
        for i in range(24):
            R_diff = torch.matmul(pred_global[:, i].transpose(-1, -2), gt_global[:, i])
            total += torch.mean(self._rotation_angle(R_diff)) * 180 / np.pi
        return total / 24

    def compute_root_rotation_error(self, pred_pose, gt_pose):
        """Root joint rotation error (degrees)."""
        F = pred_pose.shape[0]
        rot_pred = art.math.axis_angle_to_rotation_matrix(pred_pose[:, :3]).view(F, 3, 3)
        rot_gt = art.math.axis_angle_to_rotation_matrix(gt_pose[:, :3]).view(F, 3, 3)
        R_diff = torch.matmul(rot_pred.transpose(-1, -2), rot_gt)
        return torch.mean(self._rotation_angle(R_diff)) * 180 / np.pi

    @staticmethod
    def compute_rte(pred_trans, gt_trans):
        """Root Translation Error: drift between trajectory endpoints (meters)."""
        loc_pred = pred_trans[-1] - pred_trans[0]
        loc_gt = gt_trans[-1] - gt_trans[0]
        return torch.mean(torch.norm(loc_pred - loc_gt, dim=-1))

    # ------------------------------------------------------------------ #
    # Foot metrics                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_foot_slide(pred_joints_global, gt_joints_global, gt_contacts, frame_rate=100):
        """Average XY velocity error on toes/heels while in contact."""
        dt = 1 / frame_rate
        contact = gt_contacts.bool()
        # contact dims: [Frame, Left/Right, Toe/Heel] → drop frame 0 to align with diff
        masks = [
            (10, contact[1:, 0, 0]),  # left toe
            (11, contact[1:, 0, 1]),  # left heel
            (11, contact[1:, 1, 0]),  # right toe
            (12, contact[1:, 1, 1]),  # right heel
        ]

        slides = []
        for joint_idx, mask in masks:
            pred_vel = torch.diff(pred_joints_global[:, joint_idx, :2], dim=0)[mask] / dt
            gt_vel = torch.diff(gt_joints_global[:, joint_idx, :2], dim=0)[mask] / dt
            if pred_vel.shape[0] != 0:
                slides.append(torch.mean(torch.norm(pred_vel - gt_vel, dim=-1)))

        if not slides:
            raise ValueError('No foot contact found')
        return torch.stack(slides).mean()

    def compute_foot_penetration_floating_error(self, pred_joints_global, gt_joints_global,
                                                 gt_contacts, frame_rate=100, box=False):
        """Penetration / floating error of toes vs. inferred ground level (mm).

        Reference plane is bucketed by GT toe height: <0.20m → ground,
        [0.20, 0.40)m → 0.20m box, ≥0.40m → ground. The latter buckets are
        only consulted when ``box=True`` (PRISM box / stair scenes).
        """
        contact = gt_contacts.bool()
        toe_joints = [
            (10, 0, 0),  # left toe
            (11, 1, 0),  # right toe
        ]

        penetrations = []
        floatings = []
        for joint_idx, lr, th in toe_joints:
            mask = contact[:, lr, th]
            if mask.sum() == 0:
                continue
            pred_h = pred_joints_global[mask, joint_idx, 2]
            gt_h = gt_joints_global[mask, joint_idx, 2]

            buckets = [(gt_h < 0.20, 0.0)]
            if box:
                buckets.append(((gt_h >= 0.20) & (gt_h < 0.40), 0.20))
                buckets.append((gt_h >= 0.40, 0.0))  # legacy: stair vs ground

            for bucket_mask, ref in buckets:
                if bucket_mask.sum() == 0:
                    continue
                under = ref - pred_h[bucket_mask]
                over = pred_h[bucket_mask] - ref
                under = under[under > 0]
                over = over[over > 0]
                if len(under) > 0:
                    penetrations.append(under.mean())
                if len(over) > 0:
                    floatings.append(over.mean())

        penetration = torch.stack(penetrations).mean() if penetrations else torch.tensor(0.0)
        floating = torch.stack(floatings).mean() if floatings else torch.tensor(0.0)
        return penetration * self._m2mm, floating * self._m2mm

    def compute_foot_height_error_at_contact(self, pred_joints_global, gt_joints_global, gt_contacts):
        """Mean |pred_z - gt_z| on toes during contact (mm)."""
        contact = gt_contacts.bool()
        toe_joints = [(10, 0, 0), (11, 1, 0)]

        errors = []
        for joint_idx, lr, th in toe_joints:
            mask = contact[:, lr, th]
            if mask.sum() == 0:
                continue
            errors.append(torch.abs(pred_joints_global[mask, joint_idx, 2]
                                    - gt_joints_global[mask, joint_idx, 2]).mean())
        height_error = torch.stack(errors).mean() if errors else torch.tensor(0.0)
        return height_error * self._m2mm

    @staticmethod
    def compute_foot_velocity_error(pred_joints_global, gt_joints_global, frame_rate=100):
        """Average toe velocity error (m/s)."""
        dt = 1 / frame_rate
        errors = []
        for joint_idx in (10, 11):
            pred_vel = torch.diff(pred_joints_global[:, joint_idx], dim=0) / dt
            gt_vel = torch.diff(gt_joints_global[:, joint_idx], dim=0) / dt
            errors.append(torch.mean(torch.norm(pred_vel - gt_vel, dim=-1)))
        return sum(errors) / len(errors)

    @staticmethod
    def compute_grf_error(pred_grf, gt_grf):
        """Mean L2 GRF error (N). Returns 0 if predictions are missing."""
        if pred_grf is None:
            return np.zeros(1)
        return np.mean(np.linalg.norm(pred_grf - gt_grf, axis=-1))

    # ------------------------------------------------------------------ #
    # Procrustes alignment                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def batch_compute_similarity_transform_torch(S1, S2):
        """Solve the orthogonal Procrustes problem aligning S1 to S2."""
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0, 2, 1)
            S2 = S2.permute(0, 2, 1)
            transposed = True
        assert S2.shape[1] == S1.shape[1]

        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)
        K = X1.bmm(X2.permute(0, 2, 1))

        U, _, V = torch.svd(K)
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0).repeat(U.shape[0], 1, 1)
        Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))
        R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        return S1_hat.permute(0, 2, 1) if transposed else S1_hat

    # ------------------------------------------------------------------ #
    # Top-level entry                                                    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate_sequence(self, pred_pose, gt_pose, pred_trans, gt_trans,
                          pred_joints_global, gt_joints_global,
                          pred_joints_local, gt_joints_local,
                          gt_contacts, gt_grf, pred_grf, box, frame_rate=100):
        """Compute all metrics for a single (pred, gt) sequence pair."""
        # MPJPE-{Global,Local,Procrustes-Aligned}
        mpjpe_g = self._m2mm * torch.mean(torch.norm(pred_joints_global - gt_joints_global, dim=-1))
        mpjpe_l = self._m2mm * torch.mean(torch.norm(pred_joints_local - gt_joints_local, dim=-1))
        S1_hat = self.batch_compute_similarity_transform_torch(pred_joints_local, gt_joints_local)
        mpjpe_pa = self._m2mm * torch.mean(torch.norm(gt_joints_local - S1_hat, dim=-1))

        mpjre = self.compute_mpjre(pred_pose, gt_pose)
        rte = self.compute_rte(pred_trans, gt_trans)  # noqa: F841 — kept per the reference impl

        accel_pred = self._compute_acceleration(pred_joints_global, frame_rate)
        accel_gt = self._compute_acceleration(gt_joints_global, frame_rate)
        accel_err = torch.mean(torch.norm(accel_pred - accel_gt, dim=-1))

        # Jitter ratio (computed but unreported, matching reference impl)
        _ = self._compute_jitter(pred_joints_global) / self._compute_jitter(gt_joints_global)

        foot_slide = self.compute_foot_slide(pred_joints_global, gt_joints_global, gt_contacts, frame_rate)
        fp, _ = self.compute_foot_penetration_floating_error(
            pred_joints_global, gt_joints_global, gt_contacts, frame_rate, box)
        _ = self.compute_foot_height_error_at_contact(pred_joints_global, gt_joints_global, gt_contacts)
        _ = self.compute_foot_velocity_error(pred_joints_global, gt_joints_global, frame_rate)

        grf_error = self.compute_grf_error(pred_grf, gt_grf)

        metrics = {
            'MPJPE-G': mpjpe_g.detach().cpu().numpy(),
            'MPJPE-L': mpjpe_l.detach().cpu().numpy(),
            'MPJPE-PA': mpjpe_pa.detach().cpu().numpy(),
            'MPJRE': mpjre.detach().cpu().numpy(),
            'Acceleration Error': accel_err.detach().cpu().numpy(),
            'Foot Slide': foot_slide.detach().cpu().numpy(),
            'FP': fp.detach().cpu().numpy(),
            'GRF Error': grf_error,
        }
        torch.cuda.empty_cache()
        return metrics

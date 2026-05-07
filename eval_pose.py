import torch
import numpy as np
import KinematicsNet.articulate as art

class PoseEvaluator:
    def __init__(self):
        self._m2mm = 1000  # meter to millimeter
    
    # ----------------- Stability Metrics ----------------- #
    @staticmethod
    def _compute_acceleration(joints, frame_rate=100):
        """
        Calculate acceleration of joints.
        
        Args:
            joints: (F, 24, 3) joints location (m)
            frame_rate: frame rate (Hz)
        
        Returns:
            acceleration: (F-2, 24, 3) Acceleration (m/s^2)
        """
        dt = 1 / frame_rate
        velocity = (joints[1:] - joints[:-1]) / dt
        acceleration = (velocity[1:] - velocity[:-1]) / dt
        return acceleration
    
    @staticmethod
    def _compute_jitter(joints):
        """
        Calculate jitter of joints.
        
        Args:
            joints: (F, 24, 3) Joints location (m)
        
        Returns:
            jitter: average jitter value (m/s^3)
        """
        jerk = torch.diff(joints, n=3, dim=0)
        return torch.mean(torch.norm(jerk, dim=-1))
    
    
    def compute_mpjre(self, pred_pose, gt_pose):
        """
        Calculate MPJRE (Mean Per Joint Rotation Error)
        
        Args:
            pred_pose: (F, 72) predicted pose parameters
            gt_pose: (F, 72) ground truth pose parameters
        
        Returns:
            mpjre: MPJRE value (degrees)
        """
        F = pred_pose.shape[0]
        
        # Convert axis-angle representation to rotation matrices
        rot_mat_pred = art.math.axis_angle_to_rotation_matrix(pred_pose).view(F, 24, 3, 3)
        rot_mat_gt = art.math.axis_angle_to_rotation_matrix(gt_pose).view(F, 24, 3, 3)

        # Initialize global rotation matrices
        global_rot_mat_pred = torch.eye(3).to(pred_pose.device).unsqueeze(0).repeat(F, 24, 1, 1)
        global_rot_mat_gt = torch.eye(3).to(gt_pose.device).unsqueeze(0).repeat(F, 24, 1, 1)

        # Initialize rotation error accumulation variable
        total_rotation_error = 0.0

        # SMPL parent joint indices
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

        # Calculate global rotation matrices
        for i in range(24):
            if parents[i] == -1:  # Root joint
                global_rot_mat_pred[:, i] = rot_mat_pred[:, i]
                global_rot_mat_gt[:, i] = rot_mat_gt[:, i]
            else:
                parent = parents[i]
                global_rot_mat_pred[:, i] = torch.matmul(global_rot_mat_pred[:, parent], rot_mat_pred[:, i])
                global_rot_mat_gt[:, i] = torch.matmul(global_rot_mat_gt[:, parent], rot_mat_gt[:, i])

        # Calculate rotation error for each joint
        for i in range(24):
            R_diff = torch.matmul(global_rot_mat_pred[:, i].transpose(-1, -2), global_rot_mat_gt[:, i])
            angles = torch.acos(
                torch.clamp((R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2] - 1) / 2, -1 + 1e-6, 1 - 1e-6)
            )
            total_rotation_error += torch.mean(angles) * 180 / np.pi

        mpjre = total_rotation_error / 24  # Average rotation error per joint
        return mpjre
    
    def compute_root_rotation_error(self, pred_pose, gt_pose):
        """
        Calculate Root Joint Rotation Error
        
        Args:
            pred_pose: (F, 72) predicted pose parameters
            gt_pose: (F, 72) ground truth pose parameters
        
        Returns:
            root_rotation_error: Root rotation error value (degrees)
        """
        F = pred_pose.shape[0]
        
        # Extract root joint rotation (first 3 parameters)
        root_pred = pred_pose[:, :3]  # (F, 3)
        root_gt = gt_pose[:, :3]  # (F, 3)
        
        # Convert axis-angle representation to rotation matrices
        rot_mat_pred = art.math.axis_angle_to_rotation_matrix(root_pred).view(F, 3, 3)
        rot_mat_gt = art.math.axis_angle_to_rotation_matrix(root_gt).view(F, 3, 3)
        
        # Calculate rotation difference: R_diff = R_pred^T * R_gt
        R_diff = torch.matmul(rot_mat_pred.transpose(-1, -2), rot_mat_gt)
        
        # Calculate rotation angle from rotation matrix
        # Using trace formula: θ = arccos((trace(R) - 1) / 2)
        angles = torch.acos(
            torch.clamp((R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2] - 1) / 2, -1 + 1e-6, 1 - 1e-6)
        )
        
        # Convert to degrees and calculate mean
        root_rotation_error = torch.mean(angles) * 180 / np.pi
        
        return root_rotation_error
    
    def compute_rte(self, pred_trans, gt_trans):
        """
        Calculate RTE (Root Translation Error)
        
        Args:
            pred_trans: (F, 3) predicted translation parameters
            gt_trans: (F, 3) ground truth translation parameters
        
        Returns:
            rte: RTE value (meters)
        """

        # Calculate difference between trajectory endpoint and startpoint
        loc_pred = pred_trans[-1] - pred_trans[0]
        loc_gt = gt_trans[-1] - gt_trans[0]
        
        # Calculate error
        rte = torch.mean(torch.norm(loc_pred - loc_gt, dim=-1))
        return rte
    

    def compute_foot_slide(self, pred_joints_global, gt_joints_global, gt_contacts, frame_rate=100):
        """
        Calculate foot slide
        
        Args:
            pred_joints_global: (F, 24, 3) predicted global joints
            gt_joints_global: (F, 24, 3) ground truth global joints
            gt_contacts: (F, 2, 2) ground truth contacts [Frame, Left/Right, Toe/Heel]
            frame_rate: frame rate (Hz)
        """

        dt = 1 / frame_rate

        contact_bool = gt_contacts.bool()
        contact_left_toe = contact_bool[1:, 0, 0]
        contact_left_heel = contact_bool[1:, 0, 1]
        contact_right_toe = contact_bool[1:, 1, 0]
        contact_right_heel = contact_bool[1:, 1, 1]

        pred_vel_left_toe = torch.diff(pred_joints_global[:, 10, :2], dim=0)[contact_left_toe] / dt
        pred_vel_left_heel = torch.diff(pred_joints_global[:, 11, :2], dim=0)[contact_left_heel] / dt
        pred_vel_right_toe = torch.diff(pred_joints_global[:, 11, :2], dim=0)[contact_right_toe] / dt
        pred_vel_right_heel = torch.diff(pred_joints_global[:, 12, :2], dim=0)[contact_right_heel] / dt
        gt_vel_left_toe = torch.diff(gt_joints_global[:, 10, :2], dim=0)[contact_left_toe] / dt
        gt_vel_left_heel = torch.diff(gt_joints_global[:, 11, :2], dim=0)[contact_left_heel] / dt
        gt_vel_right_toe = torch.diff(gt_joints_global[:, 11, :2], dim=0)[contact_right_toe] / dt
        gt_vel_right_heel = torch.diff(gt_joints_global[:, 12, :2], dim=0)[contact_right_heel] / dt

        foot_slide_left_toe = torch.mean(torch.norm(pred_vel_left_toe - gt_vel_left_toe, dim=-1)) if pred_vel_left_toe.shape[0] != 0 else None
        foot_slide_left_heel = torch.mean(torch.norm(pred_vel_left_heel - gt_vel_left_heel, dim=-1)) if pred_vel_left_heel.shape[0] != 0 else None
        foot_slide_right_toe = torch.mean(torch.norm(pred_vel_right_toe - gt_vel_right_toe, dim=-1)) if pred_vel_right_toe.shape[0] != 0 else None
        foot_slide_right_heel = torch.mean(torch.norm(pred_vel_right_heel - gt_vel_right_heel, dim=-1)) if pred_vel_right_heel.shape[0] != 0 else None

        foot_slide_values = [
            foot_slide_left_toe, 
            foot_slide_left_heel, 
            foot_slide_right_toe, 
            foot_slide_right_heel
        ]

        valid_values = [v for v in foot_slide_values if v is not None]

        if len(valid_values) > 0:
            foot_slide_avg = torch.stack(valid_values).mean()
        else:
            raise ValueError('No foot contact found')

        return foot_slide_avg

    def compute_foot_penetration_floating_error(self, pred_joints_global, gt_joints_global, gt_contacts, frame_rate=100, box=False):
        """
        Calculate foot penetration/floating error
        
        Args:
            pred_joints_global: (F, 24, 3) predicted global joints
            gt_joints_global: (F, 24, 3) ground truth global joints
            gt_contacts: (F, 2, 2) ground truth contacts [Frame, Left/Right, Toe/Heel]
            frame_rate: frame rate (Hz)
            
        Returns:
            penetration_error: average penetration error (m)
            floating_error: average floating error (m)
        """
        contact_bool = gt_contacts.bool()
        
        # Foot joint indices: only toes
        foot_joints = [
            (10, 0, 0),  # left toe
            (11, 1, 0),  # right toe
        ]

        ground_height = 0.0
        box_small_height = 0.1903
        stair_height = 0.1958
        
        penetration_errors = []
        floating_errors = []
        
        for joint_idx, lr_idx, th_idx in foot_joints:
            # Get contact mask for this foot joint
            contact_mask = contact_bool[:, lr_idx, th_idx]
            
            if contact_mask.sum() > 0:
                # Extract height (z-coordinate) when in contact
                pred_height = pred_joints_global[contact_mask, joint_idx, 2]
                gt_height = gt_joints_global[contact_mask, joint_idx, 2]
                
                # Penetration: based on ground truth height, check against different reference planes
                # Case 1: Ground truth < 10cm -> check against ground (height = 0)
                mask_ground = gt_height < 0.20
                if mask_ground.sum() > 0:
                    penetration_ground = ground_height - pred_height[mask_ground]
                    penetration_ground = penetration_ground[penetration_ground > 0]
                    if len(penetration_ground) > 0:
                        penetration_errors.append(penetration_ground.mean())
                
                if box:
                    # Case 2: 10cm <= Ground truth < 30cm -> check against Box Small (height = 20cm)
                    mask_box = (gt_height >= 0.20) & (gt_height < 0.40)
                    if mask_box.sum() > 0:
                        penetration_box = 0.2 - pred_height[mask_box]
                        penetration_box = penetration_box[penetration_box > 0]
                        if len(penetration_box) > 0:
                            penetration_errors.append(penetration_box.mean())
                    
                    # Case 3: Ground truth >= 30cm -> check against Stair (height = 40cm)
                    mask_stair = gt_height >= 0.40
                    if mask_stair.sum() > 0:
                        penetration_stair = ground_height - pred_height[mask_stair]
                        penetration_stair = penetration_stair[penetration_stair > 0]
                        if len(penetration_stair) > 0:
                            penetration_errors.append(penetration_stair.mean())
                
                # Floating: based on ground truth height, check above different reference planes
                # Case 1: Ground truth < 10cm -> check if floating above ground (height = 0)
                if mask_ground.sum() > 0:
                    floating_ground = pred_height[mask_ground] - 0.0
                    floating_ground = floating_ground[floating_ground > 0]
                    if len(floating_ground) > 0:
                        floating_errors.append(floating_ground.mean())
                
                if box:
                    # Case 2: 10cm <= Ground truth < 30cm -> check if floating above Box Small (height = 20cm)
                    if mask_box.sum() > 0:
                        floating_box = pred_height[mask_box] - 0.2
                        floating_box = floating_box[floating_box > 0]
                        if len(floating_box) > 0:
                            floating_errors.append(floating_box.mean())
                    
                    # Case 3: Ground truth >= 30cm -> check if floating above Stair (height = 40cm)
                    if mask_stair.sum() > 0:
                        floating_stair = pred_height[mask_stair] - 0.4
                        floating_stair = floating_stair[floating_stair > 0]
                        if len(floating_stair) > 0:
                            floating_errors.append(floating_stair.mean())
        
        # Calculate average errors
        if len(penetration_errors) > 0:
            penetration_error = torch.stack(penetration_errors).mean()
        else:
            penetration_error = torch.tensor(0.0)
        
        if len(floating_errors) > 0:
            floating_error = torch.stack(floating_errors).mean()
        else:
            floating_error = torch.tensor(0.0)
        
        return penetration_error * self._m2mm, floating_error * self._m2mm


    def compute_foot_height_error_at_contact(self, pred_joints_global, gt_joints_global, gt_contacts):
        """
        Calculate foot height error during contact (simple GT comparison)
        
        Args:
            pred_joints_global: (F, 24, 3) predicted global joints
            gt_joints_global: (F, 24, 3) ground truth global joints
            gt_contacts: (F, 2, 2) ground truth contacts [Frame, Left/Right, Toe/Heel]
            
        Returns:
            height_error: average height error during contact (mm)
        """
        contact_bool = gt_contacts.bool()
        
        # Foot joint indices: only toes
        foot_joints = [
            (10, 0, 0),  # left toe
            (11, 1, 0),  # right toe
        ]
        
        height_errors = []
        
        for joint_idx, lr_idx, th_idx in foot_joints:
            # Get contact mask for this foot joint
            contact_mask = contact_bool[:, lr_idx, th_idx]
            
            if contact_mask.sum() > 0:
                # Extract height (z-coordinate) when in contact
                pred_height = pred_joints_global[contact_mask, joint_idx, 2]
                gt_height = gt_joints_global[contact_mask, joint_idx, 2]
                
                # Calculate absolute height difference
                height_diff = torch.abs(pred_height - gt_height)
                height_errors.append(height_diff.mean())
        
        # Calculate average error
        if len(height_errors) > 0:
            height_error = torch.stack(height_errors).mean()
        else:
            height_error = torch.tensor(0.0)
        
        return height_error * self._m2mm


    def compute_foot_velocity_error(self, pred_joints_global, gt_joints_global, frame_rate=100):
        """
        Calculate foot velocity error
        """
        dt = 1 / frame_rate
        pred_vel_left = torch.diff(pred_joints_global[:, 10], dim=0) / dt
        pred_vel_right = torch.diff(pred_joints_global[:, 11], dim=0) / dt
        gt_vel_left = torch.diff(gt_joints_global[:, 10], dim=0) / dt
        gt_vel_right = torch.diff(gt_joints_global[:, 11], dim=0) / dt
        foot_velocity_error_left = torch.mean(torch.norm(pred_vel_left - gt_vel_left, dim=-1))
        foot_velocity_error_right = torch.mean(torch.norm(pred_vel_right - gt_vel_right, dim=-1))
        foot_velocity_error = (foot_velocity_error_left + foot_velocity_error_right) / 2

        return foot_velocity_error


    def compute_grf_error(self, pred_grf, gt_grf):
        """
        Calculate GRF error
        """
        if pred_grf is None:
            return np.zeros(1)
        grf_error = np.mean(np.linalg.norm(pred_grf - gt_grf, axis=-1))
        return grf_error



    def batch_compute_similarity_transform_torch(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0,2,1)
            S2 = S2.permute(0,2,1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0,2,1)

        return S1_hat

    
    @torch.no_grad()
    def evaluate_sequence(self, pred_pose, gt_pose, pred_trans, gt_trans, pred_joints_global, gt_joints_global,
                            pred_joints_local, gt_joints_local, gt_contacts, gt_grf, pred_grf, box, frame_rate=100):
        """
        Evaluate the predicted sequence against the ground truth sequence.
        
        Args:
            pred_pose: (F, 72) predicted pose parameters
            gt_pose: (F, 72) ground truth pose parameters
            pred_trans: (F, 3) predicted translation parameters
            gt_trans: (F, 3) ground truth translation parameters
            pred_joints_global: (F, 24, 3) predicted global joints
            gt_joints_global: (F, 24, 3) ground truth global joints
            pred_joints_local: (F, 24, 3) predicted local joints
            gt_joints_local: (F, 24, 3) ground truth local joints
            gt_contacts: (F, 2, 2) ground truth contacts [Frame, Left/Right, Toe/Heel]
            gt_grf: (F, 2) ground truth GRF
            pred_grf: (F, 2) predicted GRF
            frame_rate: frame rate

        Returns:
            metrics: a dictionary containing all evaluation metrics
        """
        # Pre-compute SMPL outputs to avoid redundant computation
        
        # Calculate MPJPE-Global
        w_mpjpe = self._m2mm * torch.mean(torch.norm(pred_joints_global - gt_joints_global, dim=-1))

        # Calculate MPJPE-Local
        mpjpe = self._m2mm * torch.mean(torch.norm(pred_joints_local - gt_joints_local, dim=-1))
        
        # Calculate MPJPE-PA (Procrustes-aligned MPJPE)
        S1_hat = self.batch_compute_similarity_transform_torch(pred_joints_local, gt_joints_local)
        pa_mpjpe = self._m2mm * torch.mean(torch.norm(gt_joints_local - S1_hat, dim=-1))
        
        # Calculate MPJRE
        mpjre = self.compute_mpjre(pred_pose, gt_pose)
        
        # Calculate RTE (Root Translation Error)
        rte = self.compute_rte(pred_trans, gt_trans)
        
        # Calculate acceleration error
        accel_pred = self._compute_acceleration(pred_joints_global, frame_rate)
        accel_gt = self._compute_acceleration(gt_joints_global, frame_rate)
        accel_err = torch.mean(torch.norm(accel_pred - accel_gt, dim=-1))
        
        # Calculate jitter ratio
        jitter = self._compute_jitter(pred_joints_global) / self._compute_jitter(gt_joints_global)

        # Calculate foot slide
        foot_slide = self.compute_foot_slide(pred_joints_global, gt_joints_global, gt_contacts, frame_rate)

        # Foot Penetration/floating Error
        foot_penetration_error, foot_floating_error = self.compute_foot_penetration_floating_error(pred_joints_global, gt_joints_global, gt_contacts, frame_rate, box)

        # Foot Height Error at Contact
        foot_height_error = self.compute_foot_height_error_at_contact(pred_joints_global, gt_joints_global, gt_contacts)
        
        # Calculate foot velocity error
        foot_vel_error = self.compute_foot_velocity_error(pred_joints_global, gt_joints_global, frame_rate)

        # Calculate GRF error
        grf_error = self.compute_grf_error(pred_grf, gt_grf)
        
        # Compile all metrics
        metrics = {
            'MPJPE-G': w_mpjpe.detach().cpu().numpy(),
            'MPJPE-L': mpjpe.detach().cpu().numpy(),
            'MPJPE-PA': pa_mpjpe.detach().cpu().numpy(),
            'MPJRE': mpjre.detach().cpu().numpy(),
            # 'RTE': rte.detach().cpu().numpy(),
            'Acceleration Error': accel_err.detach().cpu().numpy(),
            # 'Jitter': jitter.detach().cpu().numpy(),
            'Foot Slide': foot_slide.detach().cpu().numpy(),
            'FP': foot_penetration_error.detach().cpu().numpy(),
            'FF': foot_floating_error.detach().cpu().numpy(),
            # 'FH': foot_height_error.detach().cpu().numpy(),
            # 'Foot Vel': foot_vel_error.detach().cpu().numpy(),
            'GRF Error': grf_error
        }
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        return metrics
    


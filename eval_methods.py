import os
import numpy as np
import torch
import pickle
import random
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

from aitviewer.configuration import CONFIG as C
C._conf.z_up = True
from aitviewer.models.smpl import SMPLLayer
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.scene.node import Node
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.scene.camera import PinholeCamera
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.arrows import Arrows
from aitviewer.utils import path
from aitviewer.utils.so3 import *

from eval_pose import PoseEvaluator

body_idx = {'L_Foot': 0, 'R_Foot': 1, 'L_Wrist': 2, 'R_Wrist': 3, 'Head': 4, 'Pelvis': 5, 'L_Knee': 6, 'R_Knee': 7}
imu_idx_vert = [3438, 6838, 2208, 5669, 410, 3021, 1176, 4663]
imu_idx_joint = [10, 11, 20, 21, 15, 0, 4, 5]
body_weight = 73.99543409049511

colors = {
    'gray': (149 / 255, 149 / 255, 149 / 255, 1.0),
    'blue': (102 / 255, 153 / 255, 255 / 255, 0.7),
    'green': (102 / 255, 204 / 255, 153 / 255, 0.7),
    'red': (255 / 255, 153 / 255, 153 / 255, 0.7),
    'yellow': (255 / 255, 204 / 255, 102 / 255, 0.7),
    'purple': (204 / 255, 153 / 255, 255 / 255, 0.7),
    'dark_blue': (51 / 255, 102 / 255, 153 / 255, 1.0),
    'dark_green': (102 / 255, 153 / 255, 102 / 255, 0.7),
    'dark_red': (204 / 255, 102 / 255, 102 / 255, 0.7), 
    'dark_yellow': (204 / 255, 153 / 255, 51 / 255, 0.7),
    'dark_purple': (153 / 255, 102 / 255, 153 / 255, 0.7),
}

smpl_layer = SMPLLayer(model_type="smpl", gender='male', device=C.device)

def set_smpl_sequence(pose, trans, ori, beta, gender, name='', color=(149 / 255, 149 / 255, 149 / 255, 1.0), z_up=False):
    smpl_sequence = SMPLSequence(
        poses_body=pose,
        poses_root=ori,
        betas=beta,
        trans=trans,
        is_rigged=False,
        smpl_layer=smpl_layer,
        color=color,
        z_up=z_up,
        name=f'SMPL ({name})'
    )
    return smpl_sequence


def butterworth_filter(data, cutoff=7, fs=100, order=4):
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def body_tracking_camera(targets, viewer):
    center=(0, 0, 0.5)
    radius=4
    num=2000
    start_angle=90
    end_angle=450

    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), num=num)
    c = np.column_stack((np.cos(angles) * radius, np.sin(angles) * radius, np.zeros(angles.shape)))
    circle = c + center

    num_circle = targets.shape[0] // 1000 + 1
    repeated_circle = np.tile(circle, (num_circle, 1))[:targets.shape[0]]
    positions = targets + repeated_circle
    tracking_camera = PinholeCamera(positions, targets, viewer.window_size[0], viewer.window_size[1], viewer=viewer)
    tracking_camera.name = 'Body Tracking Camera'
    viewer.scene.add(tracking_camera)
    viewer.set_temp_camera(tracking_camera)



def format_gt_data(dataset, seq_len=500):
    print('>>> Formatting GT Data')
    data_dir = f'data/{dataset}/DynamicsNet/6IMUs_Ins/test'
    data_files = sorted(os.listdir(data_dir))
    data_dict = {}
        
    for file in tqdm(data_files):
        file_name = file.split('.')[0]
        data = pickle.load(open(os.path.join(data_dir, file), 'rb'))[file_name]
        pose = data['pose_aa'][:, 3:]  # (T, 69)
        ori = data['pose_aa'][:, :3]   # (T, 3)
        trans = data['trans_orig']     # (T, 3)
        insole = data['insole_data']   # (T, 5)
        contacts = insole[:, :, 3:]    # (T, 2)
        grf = insole[:, :, 0]        # (T, 2, 1)
        grf *= body_weight 

        betas = np.zeros((pose.shape[0], 10))
        verts, joints = smpl_layer(
            poses_root=torch.from_numpy(ori).to(C.device).float(),
            poses_body=torch.from_numpy(pose).to(C.device).float(),
            betas=torch.from_numpy(betas).to(C.device).float(),
            trans=torch.from_numpy(trans).to(C.device).float(),
        )
        joints = joints[:, :24].cpu().numpy()

        if dataset == 'GRID':
            subj_id, take_id, seq_id = file_name.split('_')
        else:
            subj_id, take_id = file_name.split('_')
            seq_id = 'seq000'
        chunk_num = pose.shape[0] // seq_len
        for chunk_id in range(chunk_num):
            start_idx = chunk_id * seq_len
            end_idx = start_idx + seq_len
            pose_chunk = pose[start_idx:end_idx]
            ori_chunk = ori[start_idx:end_idx]
            trans_chunk = trans[start_idx:end_idx]
            joints_chunk = joints[start_idx:end_idx]
            contacts_chunk = contacts[start_idx:end_idx]
            grf_chunk = grf[start_idx:end_idx]
            # grf_chunk = butterworth_filter(grf_chunk, cutoff=10, fs=100, order=4)

            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 4))
            # plt.plot(grf_chunk[:, 0], label='Left Foot', linewidth=2)
            # plt.plot(grf_chunk[:, 1], label='Right Foot', linewidth=2)
            # plt.xlabel('Frame')
            # plt.ylabel('GRF (N)')
            # plt.title('Ground Reaction Force')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[f'{int(chunk_id):03d}'] = {
                'pose': pose_chunk,
                'ori': ori_chunk,
                'trans': trans_chunk,
                'joints': joints_chunk,
                'contacts': contacts_chunk,
                'grf': grf_chunk,
            }

    if dataset == 'GRID':
        ## Load Object Meshes
        obj_dir = '../MS-HPE/data/GRID/integrated'
        obj_files = os.listdir(obj_dir)
        for file in obj_files:
            if not file.endswith('.pkl'):
                continue
            with open(os.path.join(obj_dir, file), 'rb') as f:
                data = pickle.load(f)
            subj_id, take_id = file.split('.')[0].split('_')
            data_dict.setdefault(subj_id, {}).setdefault(take_id, {})['obj_meshes'] = data['objects']

    save_dir = f'output/Eval/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'gt.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)


def format_grip_results(dataset, root_dir, seq_len=500):
    print('>>> Formatting GRIP Results')
    data_dict = {}

    for mode in ['success', 'failed']:
        data_files = os.listdir(os.path.join('output/HumanoidIm/', root_dir, 'results_' + mode))
        for file in tqdm(data_files):
            data = dict(np.load(os.path.join('output/HumanoidIm/', root_dir, 'results_' + mode, file)))
            if dataset == 'GRID':
                subj_id, take_id, seq_id, chunk_id = file.split('.')[0].split('_')
            else:
                subj_id, take_id, chunk_id = file.split('.')[0].split('_')
                seq_id = 'seq000'
            pred_pos = np.concatenate((data['gt_pos'][0:2], data['pred_pos']), axis=0)
            pred_rot = np.concatenate((data['gt_rot'][0:2], data['pred_rot']), axis=0)
            disc_probs = np.concatenate((data['disc_probs'][0:2], data['disc_probs']), axis=0)
            contact_forces = np.concatenate((data['grf'][0:2], data['grf']), axis=0)  # (F, 24, 3)
            grf_left = contact_forces[:, 7, 2] + contact_forces[:, 10, 2]
            grf_right = contact_forces[:, 8, 2] + contact_forces[:, 11, 2]
            grf = np.concatenate((grf_left[:, np.newaxis], grf_right[:, np.newaxis]), axis=1)  # (F, 2)
            # grf = butterworth_filter(grf, cutoff=10, fs=100, order=4)
            
            # # Plot GRF
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 4))
            # plt.plot(grf[:, 0], label='Left Foot', linewidth=2)
            # plt.plot(grf[:, 1], label='Right Foot', linewidth=2)
            # plt.xlabel('Frame')
            # plt.ylabel('GRF (N)')
            # plt.title('Ground Reaction Force')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            F = pred_rot.shape[0]
            pred_aa = rot2aa_numpy(pred_rot.reshape(-1, 3, 3)).reshape(F, 24 * 3)
            ori = pred_aa[:, :3]
            pose = pred_aa[:, 3:]
            trans = pred_pos[:, 0]
            fail_flag = np.array([False] * F)
            # fall_frames =  (trans[:, 2] < 0.20)
            
            # fail_flag[fall_frames] = True
            # print(f'{file}: number of fall frames: {fail_flag.sum()}')
            
            betas = np.zeros((pose.shape[0], 10))
            verts, joints = smpl_layer(
                poses_root=torch.from_numpy(ori).to(C.device).float(),
                poses_body=torch.from_numpy(pose).to(C.device).float(),
                betas=torch.from_numpy(betas).to(C.device).float(),
                trans=torch.from_numpy(trans).to(C.device).float(),
            )
            joints = joints[:, :24].cpu().numpy()
            
            data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[chunk_id] = {
                'pose': pose,
                'ori': ori,
                'trans': trans,
                'joints': joints,
                'fail_flag': fail_flag,
                'grf': grf,
            }
        
    save_dir = os.path.join(f'output/Eval/{dataset}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'grip.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)


def format_globalpose_results(dataset, seq_len=500):
    print('>>> Formatting GlobalPose Results')
    data_dict = {}
    data_dir = f'../GlobalPose/output/{dataset}/inference'
    data_files = sorted(os.listdir(data_dir))
    for file in tqdm(data_files):
        if not file.endswith('.npz'):
            continue
        data = dict(np.load(os.path.join(data_dir, file)))
        if dataset == 'GRID':
            subj_id, take_id, seq_id = file.split('.')[0].split('_')
        else:
            subj_id, take_id = file.split('.')[0].split('_')
            seq_id = 'seq000'
        chunk_num = data['pose_o'].shape[0] // seq_len
        for chunk_id in range(chunk_num):
            start_idx = chunk_id * seq_len
            end_idx = start_idx + seq_len
            pred_pose = data['pose_o'][start_idx:end_idx].reshape(-1, 24*3)
            pose = pred_pose[:, 3:]
            ori = pred_pose[:, :3]
            trans = data['tran_o'][start_idx:end_idx]
            grf = data['grf_o'][start_idx:end_idx]
            grf *= body_weight
            # grf = butterworth_filter(grf, cutoff=10, fs=100, order=4)

            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 4))
            # plt.plot(grf[:, 0], label='Left Foot', linewidth=2)
            # plt.plot(grf[:, 1], label='Right Foot', linewidth=2)
            # plt.xlabel('Frame')
            # plt.ylabel('GRF (N)')
            # plt.title('Ground Reaction Force')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            betas = np.zeros((pose.shape[0], 10))
            verts, joints = smpl_layer(
                poses_root=torch.from_numpy(ori).to(C.device).float(),
                poses_body=torch.from_numpy(pose).to(C.device).float(),
                betas=torch.from_numpy(betas).to(C.device).float(),
                trans=torch.from_numpy(trans).to(C.device).float(),
            )
            joints = joints[:, :24].cpu().numpy()

            data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[f'{int(chunk_id):03d}'] = {
                'pose': pose,
                'ori': ori,
                'trans': trans,
                'joints': joints,
                'grf': grf,
            }

    save_dir = os.path.join(f'output/Eval/{dataset}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'globalpose.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)


def format_pip_results(dataset, root_dir, seq_len=500):
    print('>>> Formatting PIP Results')
    data_dict = {}
    data_dir = f'../PIP/output/{dataset}/{root_dir}/inference'

    data_files = sorted(os.listdir(data_dir))
    for file in tqdm(data_files):
        if not file.endswith('.npz'):
            continue
        data = dict(np.load(os.path.join(data_dir, file)))
        subj_id, take_id, seq_id = file.split('.')[0].split('_')

        chunk_num = data['pose_o'].shape[0] // seq_len
        for chunk_id in range(chunk_num):
            start_idx = chunk_id * seq_len
            end_idx = start_idx + seq_len
            pred_pose = data['pose_o'][start_idx:end_idx].reshape(-1, 3, 3)
            pred_pose = rot2aa_numpy(pred_pose).reshape(-1, 24 * 3)
            pose = pred_pose[:, 3:]
            ori = pred_pose[:, :3]
            trans = data['tran_o'][start_idx:end_idx]
            grf = data['grf_o'][start_idx:end_idx]
            grf *= body_weight
            # grf = butterworth_filter(grf, cutoff=10, fs=100, order=4)

            # # Plot GRF
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 4))
            # plt.plot(grf[:, 0], label='Left Foot', linewidth=2)
            # plt.plot(grf[:, 1], label='Right Foot', linewidth=2)
            # plt.xlabel('Frame')
            # plt.ylabel('GRF (N)')
            # plt.title('Ground Reaction Force')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            betas = np.zeros((pose.shape[0], 10))
            verts, joints = smpl_layer(
                poses_root=torch.from_numpy(ori).to(C.device).float(),
                poses_body=torch.from_numpy(pose).to(C.device).float(),
                betas=torch.from_numpy(betas).to(C.device).float(),
                trans=torch.from_numpy(trans).to(C.device).float(),
            )
            joints = joints[:, :24].cpu().numpy()
            
            data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[f'{int(chunk_id):03d}'] = {
                'pose': pose,
                'ori': ori,
                'trans': trans,
                'joints': joints,
                'grf': grf,
            }

    if dataset != 'GRID':
        for subj_id in data_dict.keys():
            for take_id in data_dict[subj_id].keys():
                take_data = {'seq000': {}}
                chunk_count = 0
                for seq_id in sorted(data_dict[subj_id][take_id].keys()):
                    for chunk_id in sorted(data_dict[subj_id][take_id][seq_id].keys()):
                        take_data['seq000'][f'{chunk_count:03d}'] = data_dict[subj_id][take_id][seq_id][chunk_id]
                        chunk_count += 1
                data_dict[subj_id][take_id] = take_data

    save_dir = os.path.join(f'output/Eval/{dataset}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'pip.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)


def format_mposer_results(dataset, seq_len=500):
    print('>>> Formatting MPoser Results')
    data_dict = {}
    data_dir = f'../MobilePoser/mobileposer/output/GRIP_Experiments/{dataset}'
    
    data_files = os.listdir(data_dir)
    for file in tqdm(data_files):
        if not file.endswith('.npz'):
            continue
        data = dict(np.load(os.path.join(data_dir, file)))
        subj_id, take_id, seq_id = file.split('.')[0].split('_')
        chunk_num = data['pose_o'].shape[0] // seq_len
        for chunk_id in range(chunk_num):
            start_idx = chunk_id * seq_len
            end_idx = start_idx + seq_len
            pred_pose = data['pose_o'][start_idx:end_idx]
            pred_pose = rot2aa_numpy(pred_pose).reshape(-1, 24 * 3)
            pose = pred_pose[:, 3:]
            ori = pred_pose[:, :3]
            trans = data['tran_o'][start_idx:end_idx]
            grf = data['grf_o'][start_idx:end_idx]
            grf *= body_weight
            # grf = butterworth_filter(grf, cutoff=10, fs=100, order=4)

            betas = np.zeros((pose.shape[0], 10))
            verts, joints = smpl_layer(
                poses_root=torch.from_numpy(ori).to(C.device).float(),
                poses_body=torch.from_numpy(pose).to(C.device).float(),
                betas=torch.from_numpy(betas).to(C.device).float(),
                trans=torch.from_numpy(trans).to(C.device).float(),
            )
            joints = joints[:, :24].cpu().numpy()

            data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[f'{int(chunk_id):03d}'] = {
                'pose': pose,
                'ori': ori,
                'trans': trans,
                'joints': joints,
                'grf': grf,
            }

    if dataset != 'GRID':
        for subj_id in data_dict.keys():
            for take_id in data_dict[subj_id].keys():
                take_data = {'seq000': {}}
                chunk_count = 0
                for seq_id in sorted(data_dict[subj_id][take_id].keys()):
                    for chunk_id in sorted(data_dict[subj_id][take_id][seq_id].keys()):
                        take_data['seq000'][f'{chunk_count:03d}'] = data_dict[subj_id][take_id][seq_id][chunk_id]
                        chunk_count += 1
                data_dict[subj_id][take_id] = take_data
    
    save_dir = os.path.join(f'output/Eval/{dataset}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'mposer.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)


def format_soleposer_results(dataset, seq_len=500):
    print('>>> Formatting SolePoser Results')
    data_dict = {}
    data_dir = f'../SolePoser/results/{dataset}'
    
    data_files = os.listdir(data_dir)
    for file in tqdm(data_files):
        if not file.endswith('.npz'):
            continue
        data = dict(np.load(os.path.join(data_dir, file)))
        subj_id, take_id, seq_id = file.split('.')[0].split('_')
        chunk_num = data['pred_pose'].shape[0] // seq_len
        for chunk_id in range(chunk_num):
            start_idx = chunk_id * seq_len
            end_idx = start_idx + seq_len
            joints = data['pred_pose'][start_idx:end_idx]

            seq_str = seq_id if dataset == 'GRID' else f'seq{seq_id}'
            data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_str, {})[f'{int(chunk_id):03d}'] = {
                'pose': None,
                'ori': None,
                'trans': None,
                'joints': joints,
            }

    if dataset != 'GRID':
        for subj_id in data_dict.keys():
            for take_id in data_dict[subj_id].keys():
                take_data = {'seq000': {}}
                chunk_count = 0
                for seq_id in sorted(data_dict[subj_id][take_id].keys()):
                    for chunk_id in sorted(data_dict[subj_id][take_id][seq_id].keys()):
                        take_data['seq000'][f'{chunk_count:03d}'] = data_dict[subj_id][take_id][seq_id][chunk_id]
                        chunk_count += 1
                data_dict[subj_id][take_id] = take_data
    
    save_dir = os.path.join(f'output/Eval/{dataset}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'soleposer.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)


def format_form_results(dataset, seq_len=500):
    print('>>> Formatting Form Results')
    data_dict = {}
    data_dir = f'../FoRM/output/{dataset}'
    
    data_files = os.listdir(data_dir)
    for file in tqdm(data_files):
        data = dict(pickle.load(open(os.path.join(data_dir, file), 'rb')))
        subj_id, take_id, seq_id = file.split('.')[0].split('_')
        chunk_num = data['pred_poses'].shape[0] // seq_len
        for chunk_id in range(chunk_num):
            start_idx = chunk_id * seq_len
            end_idx = start_idx + seq_len
            pose = data['pred_poses'][start_idx:end_idx, 3:].numpy()
            ori = data['pred_poses'][start_idx:end_idx, :3].numpy()

            # fix discontinuity problem in orientation
            zero_indices = np.where(np.any(ori == 0.0, axis=1))
            if len(zero_indices) > 0:
                for index in zero_indices:
                    ori[index] = ori[index - 1]

            trans = data['pred_trans'][start_idx:end_idx].numpy()
            betas = np.zeros((pose.shape[0], 10))
            verts, joints = smpl_layer(
                poses_root=torch.from_numpy(ori).to(C.device).float(),
                poses_body=torch.from_numpy(pose).to(C.device).float(),
                betas=torch.from_numpy(betas).to(C.device).float(),
                trans=torch.from_numpy(trans).to(C.device).float(),
            )
            joints = joints[:, :24].cpu().numpy()

            data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[f'{int(chunk_id):03d}'] = {
                'pose': pose,
                'ori': ori,
                'trans': trans,
                'joints': joints,
            }

    if dataset != 'GRID':
        for subj_id in data_dict.keys():
            for take_id in data_dict[subj_id].keys():
                take_data = {'seq000': {}}
                chunk_count = 0
                for seq_id in sorted(data_dict[subj_id][take_id].keys()):
                    for chunk_id in sorted(data_dict[subj_id][take_id][seq_id].keys()):
                        take_data['seq000'][f'{chunk_count:03d}'] = data_dict[subj_id][take_id][seq_id][chunk_id]
                        chunk_count += 1
                data_dict[subj_id][take_id] = take_data
    
    save_dir = os.path.join(f'output/Eval/{dataset}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'form.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)


def calculate_zmp_distance(dataset):
    from eval_zmp import evaluate_zmp_distance
    print('>>> Calculating ZMP Distance')
    root_dir = f'output/Eval/{dataset}'

    data_path = os.path.join(root_dir, f'gt.pkl')
    data = pickle.load(open(data_path, 'rb'))
    contacts_dict = {}

    for subj_id in sorted(data.keys()):
        for take_id in sorted(data[subj_id].keys()):
            for seq_id in sorted(data[subj_id][take_id].keys()):
                if 'seq' not in seq_id:
                    continue
                for chunk_id in sorted(data[subj_id][take_id][seq_id].keys()):
                    contacts = data[subj_id][take_id][seq_id][chunk_id]['contacts']
                    contacts_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[chunk_id] = contacts

    for method in ['GT', 'GRIP', 'GlobalPose', 'PIP', 'MPoser', 'FoRM']:
        print(f'>>> Processing {method}')
        data_path = os.path.join(root_dir, f'{method.lower()}.pkl')
        data = pickle.load(open(data_path, 'rb'))
        zmp_dict = {}

        for subj_id in sorted(data.keys()):
            for take_id in sorted(data[subj_id].keys()):
                for seq_id in sorted(data[subj_id][take_id].keys()):
                    if 'seq' not in seq_id:
                        continue
                    for chunk_id in sorted(data[subj_id][take_id][seq_id].keys()):
                        print(f'    {subj_id}_{take_id}_{seq_id}_{chunk_id}', end='\r')
                        pose = data[subj_id][take_id][seq_id][chunk_id]['pose']
                        trans = data[subj_id][take_id][seq_id][chunk_id]['trans']
                        ori = data[subj_id][take_id][seq_id][chunk_id]['ori']
                        ori = aa2rot_numpy(ori).reshape(-1, 1, 3, 3)

                        # Convert to y-up coordinate system
                        if method != 'PIP' and method != 'MPoser':
                            R_y_z = euler2rot_numpy(np.array([90, 180, 0]), degrees=True)
                            R_z_y = R_y_z.T
                            ori = R_z_y @ ori
                            trans = trans @ R_z_y.T
                        
                        pose = aa2rot_numpy(pose.reshape(-1, 3)).reshape(-1, 23, 3, 3)
                        poses = np.concatenate((ori, pose), axis=1) # [F, 24, 3, 3]
                        contacts = contacts_dict[subj_id][take_id][seq_id][chunk_id].numpy().astype(np.bool_)
                        contacts = contacts[:, :, 0] | contacts[:, :, 1]
                        zmp, dist = evaluate_zmp_distance(poses, trans, contacts)
                        zmp_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[chunk_id] = {
                            'ZMP': zmp,    # [F, 3]
                            'Dist': dist,  # [F]
                        }

        save_dir = os.path.join(f'output/Eval/{dataset}')
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'{method.lower()}_zmp.pkl'), 'wb') as f:
            pickle.dump(zmp_dict, f)
    print(f'\n>>> Done')


def integrate_results(dataset):
    print('>>> Integrating Results')
    data_dict = {'seq_ids': set()}
    root_dir = f'output/Eval/{dataset}'
    R_y_z = euler2rot_numpy(np.array([90, 180, 0]), degrees=True)

    for method in ['GT', 'GRIP', 'GlobalPose', 'PIP', 'MPoser', 'SolePoser', 'FoRM']:
        data_path = os.path.join(root_dir, f'{method.lower()}.pkl')
        data = pickle.load(open(data_path, 'rb'))
        if method != 'SolePoser':
            zmp_path = os.path.join(root_dir, f'{method.lower()}_zmp.pkl')
            zmp_data = pickle.load(open(zmp_path, 'rb'))
            zmp_dists = []

        for subj_id in sorted(data.keys()):
            data_dict.setdefault(subj_id, {})
            for take_id in sorted(data[subj_id].keys()):
                data_dict[subj_id].setdefault(take_id, {})

                ## Load Object Meshes
                if method == 'GT' and dataset == 'GRID':
                    obj_meshes = data[subj_id][take_id]['obj_meshes']
                    data_dict.setdefault(subj_id, {}).setdefault(take_id, {})['obj_meshes'] = obj_meshes
                
                ## Load SMPL Params
                for seq_id in sorted(data[subj_id][take_id].keys()):
                    data_dict[subj_id][take_id].setdefault(seq_id, {})

                    if seq_id == 'obj_meshes':
                        continue

                    for chunk_id in sorted(data[subj_id][take_id][seq_id].keys()):
                        seq_label = f'{subj_id}_{take_id}_{seq_id}_{chunk_id}'
                        # if 'subj001_take022_seq003_001' not in seq_label:
                        #     continue
                        data_dict[subj_id][take_id][seq_id].setdefault(chunk_id, {}).setdefault(method, {})

                        pose = data[subj_id][take_id][seq_id][chunk_id]['pose']
                        trans = data[subj_id][take_id][seq_id][chunk_id]['trans']
                        ori = data[subj_id][take_id][seq_id][chunk_id]['ori']
                        joints = data[subj_id][take_id][seq_id][chunk_id]['joints']
                        if method != 'SolePoser':
                            zmp = zmp_data[subj_id][take_id][seq_id][chunk_id]['ZMP']
                            dist = zmp_data[subj_id][take_id][seq_id][chunk_id]['Dist']
                            zmp = zmp @ R_y_z.T
                            zmp_dists.append(dist)
                        else:
                            zmp = None
                            dist = None

                        if method == 'PIP' or method == 'MPoser':
                            root_offset = trans[0] - joints[0, 0]
                            joints = joints + root_offset
                            ori = rot2aa_numpy(R_y_z @ aa2rot_numpy(ori))
                            trans = trans @ R_y_z.T
                            joints = joints @ R_y_z.T - root_offset

                        if method != 'GT' and method != 'SolePoser':
                            trans_gt = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['trans']
                            joints_gt = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['joints']
                            contacts = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['contacts']
                            contacts_mask = contacts[:, :, 0].bool() | contacts[:, :, 1].bool()

                            trans_dff = trans_gt[0] - trans[0]
                            # if method == 'GRIP':
                            #     trans_dff += np.array([0, 0, 0.02])
                                
                            trans[:, :2] += trans_dff[:2]
                            joints[:, :, :2] += trans_dff.reshape(1, 1, 3).repeat(24, axis=1).repeat(joints.shape[0], axis=0)[:, :, :2]
                            zmp = zmp + np.array([trans_dff[0], trans_dff[1], 0])

                            if method == 'GlobalPose' or method == 'PIP' or method == 'MPoser':
                                foot_joints_height = joints[:, [10, 11], 2]
                                foot_joints_height_gt = joints_gt[:, [10, 11], 2]
                                foot_joints_height = foot_joints_height[contacts_mask]
                                foot_joints_height_gt = foot_joints_height_gt[contacts_mask]

                                hist, bin_edges = np.histogram(foot_joints_height.flatten(), bins=50)
                                most_frequent_bin_idx = np.argmax(hist)
                                most_frequent_height = (bin_edges[most_frequent_bin_idx] + bin_edges[most_frequent_bin_idx + 1]) / 2

                                trans[:, 2] -= most_frequent_height
                                joints[:, :, 2] -= most_frequent_height
                            elif method == 'FoRM':
                                trans[:, 2] += trans_dff[2]
                                joints[:, :, 2] += trans_dff.reshape(1, 1, 3).repeat(24, axis=1).repeat(joints.shape[0], axis=0)[:, :, 2]


                        if method != 'SolePoser' and method != 'FoRM':
                            grf = data[subj_id][take_id][seq_id][chunk_id]['grf']
                            grf = butterworth_filter(grf, cutoff=7, fs=100, order=4)
                        else:
                            grf = None

                        data_dict[subj_id][take_id][seq_id][chunk_id][method] = {
                            'pose': pose,
                            'trans': trans,
                            'ori': ori,
                            'joints': joints,
                            'zmp': zmp,
                            'dist': dist,
                            'grf': grf,
                        }
                        if method == 'GT':
                            contacts = data[subj_id][take_id][seq_id][chunk_id]['contacts']
                            data_dict[subj_id][take_id][seq_id][chunk_id][method]['contacts'] = contacts

                        if method == 'GRIP':
                            fail_flag = data[subj_id][take_id][seq_id][chunk_id]['fail_flag']
                            data_dict[subj_id][take_id][seq_id][chunk_id]['fail_flag'] = fail_flag
                        data_dict['seq_ids'].add(seq_label)
        # if method != 'SolePoser':
        #     print(f'{method} ZMP Dist: {np.mean(zmp_dists)}')

    ## Filter out sequences that do not contain all methods
    new_seq_ids = []
    new_data_dict = {}
    for seq_id in data_dict['seq_ids']:
        subj_id, take_id, seq_id, chunk_id = seq_id.split('_')
        data_method_keys = set(data_dict[subj_id][take_id][seq_id][chunk_id].keys())
        all_method_keys = set(['GT', 'GRIP', 'GlobalPose', 'PIP', 'MPoser', 'SolePoser', 'FoRM'])
        if all(method in data_method_keys for method in all_method_keys):
            new_seq_ids.append(f'{subj_id}_{take_id}_{seq_id}_{chunk_id}')
            new_data_dict.setdefault(subj_id, {}).setdefault(take_id, {}).setdefault(seq_id, {})[chunk_id] = data_dict[subj_id][take_id][seq_id][chunk_id]
            new_data_dict[subj_id][take_id][seq_id][chunk_id]['fail_flag'] = data_dict[subj_id][take_id][seq_id][chunk_id]['fail_flag']
            if 'obj_meshes' in data_dict[subj_id][take_id].keys():
                new_data_dict[subj_id][take_id]['obj_meshes'] = data_dict[subj_id][take_id]['obj_meshes']
        else:
            continue
            # print(f'Skipping {subj_id}_{take_id}_{seq_id}_{chunk_id} because it does not contain all methods')

    new_data_dict['seq_ids'] = new_seq_ids
    del data_dict

    ## Save Results
    save_dir = os.path.join(f'output/Eval/{dataset}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'integrated.pkl'), 'wb') as f:
        pickle.dump(new_data_dict, f)



def evaluate_results(dataset):
    print('>>> Evaluating Results')
    data_dict = pickle.load(open(f'output/Eval/{dataset}/integrated.pkl', 'rb'))
    seq_labels = data_dict['seq_ids']
    error_dict = {'MPJPE-G': {}, 'MPJPE-L': {}, 'MPJPE-PA': {}, 'MPJRE': {}, 'Acceleration Error': {}, 'Foot Slide': {}, 'FP': {}, 'FF': {}, 'FH': {}, 'GRF Error': {}}
    pose_evaluator = PoseEvaluator()
    box = True if dataset == 'GRID' else False

    # fp_seqs = {}

    for seq_label in tqdm(seq_labels):
        # if 'subj002_take017_seq001_001' not in seq_label:
        #     continue
        subj_id, take_id, seq_id, chunk_id = seq_label.split('_')
        fail_flag = data_dict[subj_id][take_id][seq_id][chunk_id]['fail_flag']
        # if fail_flag.any():
        #     continue
        pose_gt = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['pose']
        trans_gt = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['trans']
        ori_gt = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['ori']
        poses_gt = np.concatenate((ori_gt, pose_gt), axis=1)  # [F, 72]
        joints_gt_global = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['joints']
        joints_gt_local = joints_gt_global - joints_gt_global[:, 0:1]
        zmp_dist_gt = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['dist']
        contacts_gt = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['contacts']
        grf_gt = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['grf']

        for method in ['GRIP', 'GlobalPose', 'PIP', 'MPoser', 'FoRM']:
            pose_pred = data_dict[subj_id][take_id][seq_id][chunk_id][method]['pose']
            trans_pred = data_dict[subj_id][take_id][seq_id][chunk_id][method]['trans']
            ori_pred = data_dict[subj_id][take_id][seq_id][chunk_id][method]['ori']
            poses_pred = np.concatenate((ori_pred, pose_pred), axis=1).astype(np.float32)
            joints_pred_global = data_dict[subj_id][take_id][seq_id][chunk_id][method]['joints']
            joints_pred_local = joints_pred_global - joints_pred_global[:, 0:1]
            grf_pred = data_dict[subj_id][take_id][seq_id][chunk_id][method]['grf']

            metrics = pose_evaluator.evaluate_sequence(
                torch.tensor(poses_pred).float(), torch.tensor(poses_gt).float(), 
                torch.tensor(trans_pred).float(), torch.tensor(trans_gt).float(), 
                torch.tensor(joints_pred_global).float(), torch.tensor(joints_gt_global).float(), 
                torch.tensor(joints_pred_local).float(), torch.tensor(joints_gt_local).float(),
                contacts_gt, grf_gt, grf_pred, box)
            
            # if method == 'GRIP' and metrics['FP'] > 0:
            #     fp_seqs[f'{subj_id}_{take_id}_{seq_id}_{chunk_id}'] = metrics['FP']

            zmp_dist_pred = data_dict[subj_id][take_id][seq_id][chunk_id][method]['dist']
            zmp_error = np.sqrt((zmp_dist_pred - zmp_dist_gt) ** 2)
            # metrics['ZMP Error'] = zmp_error

            for key, value in metrics.items():
                error_dict[key].setdefault(method, [])
                error_dict[key][method].append(value)
        
        joints_pred_local = data_dict[subj_id][take_id][seq_id][chunk_id]['SolePoser']['joints'] - data_dict[subj_id][take_id][seq_id][chunk_id]['SolePoser']['joints'][:, 0:1]
        joints_gt_local = joints_gt_local[:, [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21]]  # Extract 17 joints
        joints_pred_local = torch.tensor(joints_pred_local).float()
        joints_gt_local = torch.tensor(joints_gt_local).float()
        S1_hat = pose_evaluator.batch_compute_similarity_transform_torch(joints_pred_local, joints_gt_local)
        pa_mpjpe = np.mean(np.linalg.norm(joints_gt_local - S1_hat, axis=-1)) * 1000  # m to mm
        error_dict['MPJPE-PA'].setdefault('SolePoser', [])
        error_dict['MPJPE-PA']['SolePoser'].append(pa_mpjpe)
        accel_pred = pose_evaluator._compute_acceleration(joints_pred_global, 100)
        accel_gt = pose_evaluator._compute_acceleration(joints_gt_global, 100)
        accel_err = np.mean(np.linalg.norm(accel_pred - accel_gt, axis=-1))
        error_dict['Acceleration Error'].setdefault('SolePoser', [])
        error_dict['Acceleration Error']['SolePoser'].append(accel_err)

    print('\n' + '='*200)
    print('Evaluation Results')
    print('='*200)
    print(f"{'Method':<15} {'MPJPE-G (mm)':<15} {'MPJPE-L (mm)':<15} {'MPJPE-PA (mm)':<15} {'MPJRE (deg)':<15} {'Acc (m/s^2)':<15} {'FS (m/s)':<15} {'FP (mm)':<15} {'FF (mm)':<15} {'GRF Error (N)':<15}")
    print('-'*200)
    
    # Prepare data for Excel export
    results_data = []
    
    for method in ['GRIP', 'GlobalPose', 'PIP', 'MPoser', 'FoRM']:
        mpjpe_global = np.mean(error_dict['MPJPE-G'][method])
        mpjpe_local = np.mean(error_dict['MPJPE-L'][method])
        pa_mpjpe = np.mean(error_dict['MPJPE-PA'][method])
        mpjre = np.mean(error_dict['MPJRE'][method])
        # rte = np.mean(error_dict['RTE'][method])
        acc_error = np.mean(error_dict['Acceleration Error'][method])
        # jitter = np.mean(error_dict['Jitter'][method])
        foot_slide = np.mean(error_dict['Foot Slide'][method])
        fp = np.mean(error_dict['FP'][method])
        ff = np.mean(error_dict['FF'][method])
        # zmp_error = np.mean(error_dict['ZMP Error'][method])
        grf_error = np.mean(error_dict['GRF Error'][method])
        print(f'{method:<15} {mpjpe_global:<15.2f} {mpjpe_local:<15.2f} {pa_mpjpe:<15.2f} {mpjre:<15.2f} {acc_error:<15.2f} {foot_slide:<15.2f} {fp:<15.2f} {ff:<15.2f} {grf_error:<15.2f}')
        results_data.append({
            'Method': method,
            'MPJPE-G (mm)': mpjpe_global,
            'MPJPE-L (mm)': mpjpe_local,
            'MPJPE-PA (mm)': pa_mpjpe,
            'MPJRE (deg)': mpjre,
            # 'RTE (m)': rte,
            'Acc Error (m/s^2)': acc_error,
            # 'Jitter': jitter,
            'Foot Slide (m/s)': foot_slide,
            'FP (mm)': fp,
            'FF (mm)': ff,
            # 'ZMP Error (m)' : zmp_error,
            'GRF Error (N)': grf_error
        })
        
    
    pa_mpjpe = np.mean(error_dict['MPJPE-PA']['SolePoser'])
    print(f"{'SolePoser':<15} {'N/A':<15} {'N/A':<15} {pa_mpjpe:<15.2f} {'N/A':<15} {accel_err:<15.2f} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    results_data.append({
        'Method': 'SolePoser',
        'MPJPE-G (mm)': 'N/A',
        'MPJPE-L (mm)': 'N/A',
        'MPJPE-PA (mm)': pa_mpjpe,
        'MPJRE (deg)': 'N/A',
        # 'RTE (m)': 'N/A',
        'Acc Error (m/s^2)': accel_err,
        # 'Jitter': 'N/A',
        'Foot Slide (m/s)': 'N/A',
        'FP (mm)': 'N/A',
        'FF (mm)': 'N/A',
        # 'ZMP Error (m)': 'N/A',
        'GRF Error (N)': 'N/A'
    })
    
    print('='*200 + '\n')
    
    # Save results to Excel
    df = pd.DataFrame(results_data)
    excel_path = f'output/Eval/{dataset}/evaluation_results.xlsx'
    df.to_excel(excel_path, index=False, sheet_name='Results')
    print(f'Results saved to: {excel_path}')

    # for seq_id, fp in fp_seqs.items():
    #     print(f'{seq_id} FP: {fp}')


def vis_contact(smpl_sequence, contacts):
    mesh_ids = {'LF': 3222, 'LB': 3386, 'RF': 6620, 'RB': 6787}
    contacts_left = contacts[:, 0]  # [F, 2]
    contacts_left_front = contacts_left[:, 0]
    contacts_left_back = contacts_left[:, 1]
    F = contacts_left.shape[0]

    contacts_right = contacts[:, 1]  # [F, 2]
    contacts_right_front = contacts_right[:, 0]
    contacts_right_back = contacts_right[:, 1]

    contact_pcs = Node(name='Foot Contact')

    left_front_pos = smpl_sequence.vertices[:, mesh_ids['LF']]
    colors = np.array([(1.0, 0.0, 0.0, 1.0)] * F)
    colors[contacts_left_front == 1] = (0.0, 1.0, 0.0, 1.0)
    pc_left_front = PointClouds(left_front_pos, color=(1.0, 0.0, 0.0, 1.0), point_size=20.0, name='Left Front', colors=colors)

    left_back_pos = smpl_sequence.vertices[:, mesh_ids['LB']]
    colors = np.array([(1.0, 0.0, 0.0, 1.0)] * F)
    colors[contacts_left_back == 1] = (0.0, 1.0, 0.0, 1.0)
    pc_left_back = PointClouds(left_back_pos, color=(1.0, 0.0, 0.0, 1.0), point_size=20.0, name='Left Back', colors=colors)

    right_front_pos = smpl_sequence.vertices[:, mesh_ids['RF']]
    colors = np.array([(1.0, 0.0, 0.0, 1.0)] * F)
    colors[contacts_right_front == 1] = (0.0, 1.0, 0.0, 1.0)
    pc_right_front = PointClouds(right_front_pos, color=(1.0, 0.0, 0.0, 1.0), point_size=20.0, name='Right Front', colors=colors)

    right_back_pos = smpl_sequence.vertices[:, mesh_ids['RB']]
    colors = np.array([(1.0, 0.0, 0.0, 1.0)] * F)
    colors[contacts_right_back == 1] = (0.0, 1.0, 0.0, 1.0)
    pc_right_back = PointClouds(right_back_pos, color=(1.0, 0.0, 0.0, 1.0), point_size=20.0, name='Right Back', colors=colors)

    contact_pcs.add(pc_left_front, pc_left_back, pc_right_front, pc_right_back)
    return contact_pcs


def visualize_results(dataset, headless=False):
    print('>>> Visualizing Results')
    method_color = {
        'GT': colors['gray'],
        'GRIP': colors['dark_blue'],
        'GlobalPose': colors['dark_purple'],
        'PIP': colors['dark_green'],
        'MPoser': colors['dark_red'],
        'FoRM': colors['dark_yellow'],
    }
    
    data_dict = pickle.load(open(f'output/Eval/{dataset}/integrated.pkl', 'rb'))
    seq_labels = sorted(data_dict['seq_ids'])

    if headless:
        video_dir = f'output/Eval/{dataset}/visualization'
        os.makedirs(video_dir, exist_ok=True)
        viewer = HeadlessRenderer()

    # random.shuffle(seq_labels)
    for seq_label in seq_labels:
        if 'subj009_take013_seq000_034' not in seq_label:
                continue
        subj_id, take_id, seq_id, chunk_id = seq_label.split('_')
        print(f'>>> Sequence: {subj_id}_{take_id}_{seq_id}_{chunk_id}')

        ## Root-relative Joints
        joints_seq = {}
        for method in ['GT', 'GRIP', 'GlobalPose', 'PIP', 'MPoser', 'FoRM']:
            joints = data_dict[subj_id][take_id][seq_id][chunk_id][method]['joints']
            joints_seq[method] = Spheres(positions=joints, radius=0.03, color=method_color[method], name=f'Joints ({method})')
            # pose = data_dict[subj_id][take_id][seq_id][chunk_id][method]['pose']
            # smpl = set_smpl_sequence(pose=pose, trans=None, ori=None, beta=None, gender='male', name=method, color=method_color[method])
            # joints_seq[method] = Spheres(positions=smpl.joints, radius=0.03, color=method_color[method], name=f'Joints ({method})')
        joints = data_dict[subj_id][take_id][seq_id][chunk_id]['SolePoser']['joints']  # [F, 17, 3]
        joints_seq['SolePoser'] = Spheres(positions=joints, radius=0.03, color=colors['dark_yellow'], name=f'Joints (SolePoser)')


        ## SMPL Sequences
        smpl_seq = {}
        for method in ['GT', 'GRIP', 'GlobalPose', 'PIP', 'MPoser', 'FoRM']:
            pose = data_dict[subj_id][take_id][seq_id][chunk_id][method]['pose']
            trans = data_dict[subj_id][take_id][seq_id][chunk_id][method]['trans']
            ori = data_dict[subj_id][take_id][seq_id][chunk_id][method]['ori']
            smpl_seq[method] = set_smpl_sequence(pose=pose, trans=trans, ori=ori, beta=None, gender='male', name=method, color=method_color[method])

        ## ZMP
        zmp_seq = {}
        for method in ['GT', 'GRIP', 'GlobalPose', 'PIP', 'MPoser', 'FoRM']:
            zmp = data_dict[subj_id][take_id][seq_id][chunk_id][method]['zmp'].reshape(-1, 1, 3)
            zmp_seq[method] = Spheres(positions=zmp, radius=0.03, color=method_color[method], name=f'ZMP ({method})')


        objects = data_dict[subj_id][take_id]['obj_meshes'] if dataset == 'GRID' else None
        # Create Camera
        target = smpl_seq['GT'].joints[:, 0]
        target[:, 2] = 1
        target = butterworth_filter(target, cutoff=0.5)


        ## Contact
        contacts = data_dict[subj_id][take_id][seq_id][chunk_id]['GT']['contacts']
        contact_pcs = vis_contact(smpl_seq['GT'], contacts)
        

        ## Visualization
        if not headless:
            viewer = Viewer()
        viewer.scene.add(smpl_seq['GT'])
        viewer.scene.add(smpl_seq['GRIP'])
        viewer.scene.add(smpl_seq['GlobalPose'])
        viewer.scene.add(smpl_seq['PIP'])
        viewer.scene.add(smpl_seq['MPoser'])
        viewer.scene.add(smpl_seq['FoRM'])
        # viewer.scene.add(joints_seq['SolePoser'])
        # viewer.scene.add(joints_seq['GT'])
        viewer.scene.add(joints_seq['GRIP'])
        # viewer.scene.add(joints_seq['GlobalPose'])
        # viewer.scene.add(joints_seq[ 
        # viewer.scene.add(joints_seq['FoRM'])
        # viewer.scene.add(zmp_seq['GT'])
        # viewer.scene.add(zmp_seq['GRIP'])
        # viewer.scene.add(zmp_seq['GlobalPose'])
        # viewer.scene.add(zmp_seq['PIP'])
        # viewer.scene.add(zmp_seq['MPoser'])
        # viewer.scene.add(zmp_seq['FoRM'])
        
        viewer.scene.add(contact_pcs)

        if objects is not None:
            for obj_name, obj_data in objects.items():
                face_colors = np.ones((1, obj_data['faces'].shape[0], 4)) * np.array([0.5, 0.5, 0.5, 1.0])
                viewer.scene.add(Meshes(vertices=obj_data['vertices'], faces=obj_data['faces'], name=obj_name, face_colors=face_colors))

        body_tracking_camera(target, viewer)

        ## Visualization Settings
        viewer.auto_set_floor = False
        viewer.scene.floor.enabled = True
        viewer.scene.origin.enabled = True
        viewer.scene.fps = 30.0
        viewer.playback_fps = 100.0
        viewer.shadows_enabled = True
        viewer.auto_set_camera_target = False

        if headless:
            video_path = os.path.join(video_dir, f'{subj_id}_{take_id}_{seq_id}_{chunk_id}.mp4')
            viewer.save_video(video_dir=video_path, ensure_no_overwrite=False)
            viewer.reset()
        else:
            viewer.run()
            viewer.close()
        gc.collect()
        torch.cuda.empty_cache()

    # viewer = HeadlessRenderer()

if __name__ == '__main__':

    ## Step 1: Format Data

    # dataset = 'GRID'  # 'GRID', 'UnderPressure', 'PSU-TMM100'
    # format_gt_data(dataset)
    # format_grip_results(dataset, 'CVPR_GRID_prim0_4IMUs')
    # format_globalpose_results(dataset)
    # format_pip_results(dataset, '20251024_101426')
    # format_mposer_results(dataset)
    # format_form_results(dataset)
    # format_soleposer_results(dataset)
    # calculate_zmp_distance(dataset)  # Require rdbl env

    # dataset = 'UnderPressure'  # 'GRID', 'UnderPressure', 'PSU-TMM100'
    # format_gt_data(dataset)
    # format_grip_results(dataset, 'CVPR_UP_prim0_4IMUs')
    # format_globalpose_results(dataset)
    # format_pip_results(dataset, '20251024_101334')
    # format_mposer_results(dataset)
    # format_form_results(dataset)
    # format_soleposer_results(dataset)
    # calculate_zmp_distance(dataset)  # Require rdbl env

    dataset = 'PSU-TMM100'  # 'GRID', 'UnderPressure', 'PSU-TMM100'
    # format_gt_data(dataset)
    # format_grip_results(dataset, 'CVPR_PSU_prim0_4IMUs')
    # format_globalpose_results(dataset)
    # format_pip_results(dataset, '20251024_101341')
    # format_mposer_results(dataset)
    # format_form_results(dataset)
    # format_soleposer_results(dataset)
    # calculate_zmp_distance(dataset)  # Require rdbl env

    ## Step 2: Integrate Results
    integrate_results(dataset)


    ## Stage 3: Evaluate Results
    evaluate_results(dataset)


    ## Step 4: Visualize Results
    visualize_results(dataset)
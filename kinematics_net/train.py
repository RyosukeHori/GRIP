"""Train KinematicsNet either one sub-network at a time (``--mode independent``)
or end-to-end (``--mode joint``).

Independent training freezes the other sub-networks; joint training fine-tunes
the whole pipeline. Run independent first to populate ``{save_dir}/models/{PL,PA,VL,RA}.pt``
then joint training to obtain ``{save_dir}/models/best_model.pt``.
"""
from __future__ import annotations  # required for PEP 585/604 syntax on Python 3.8

import argparse
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb

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
# Hyperparameters                                                             #
# --------------------------------------------------------------------------- #
# Edit these dicts to tune training; ``init_wandb`` logs whichever config is
# active for the current run, so the wandb dashboard always matches the values
# actually used below.

INDEPENDENT_HPARAMS = {
    'batch_size': 512,
    'learning_rate': 1e-4,
    'num_epochs': 300,
    'sequence_length': 100,
    'overlap': 0,
}

JOINT_HPARAMS = {
    'batch_size': 256,
    'learning_rate': 1e-5,
    'num_epochs': 2000,
    'sequence_length': 500,
    'overlap': 50,
}

# Per-term weights used by KinematicsNetLoss during joint training.
JOINT_LOSS_WEIGHTS = {
    'pl_weight': 1.0,
    'pa_weight': 1.0,
    'vl_weight': 1.0,
    'ra_weight': 1.0,
}


# --------------------------------------------------------------------------- #
# Sub-network registry                                                        #
# --------------------------------------------------------------------------- #

# Mapping from network short name to the attribute on KinematicsNet.
SUBMODULE_ATTR = {
    'PL': 'plnet',
    'PA': 'panet',
    'VL': 'vlnet',
    'RA': 'ranet',
}

# What each sub-network requires to have been trained first.
DEPENDENCIES = {
    'PL': [],
    'PA': ['PL'],
    'VL': ['PL', 'PA'],
    'RA': ['PL', 'PA', 'VL'],
}

NETWORK_ORDER = ['PL', 'PA', 'VL', 'RA']


def _submodule(model: KinematicsNet, name: str) -> nn.Module:
    return getattr(model, SUBMODULE_ATTR[name])


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

def init_wandb(use_wandb: bool, project_name: str, run_name: Optional[str],
               config: dict) -> bool:
    """Start a wandb run, logging ``config`` as the run's hyperparameters."""
    if not use_wandb:
        return False
    wandb.init(
        project=project_name,
        name=run_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
        config=config,
    )
    return True


def log_metrics(use_wandb: bool, metrics: dict, step: Optional[int] = None) -> None:
    if use_wandb:
        wandb.log(metrics, step=step) if step is not None else wandb.log(metrics)
    else:
        for key, value in metrics.items():
            print(f"{key}: {value:.6f}")


# --------------------------------------------------------------------------- #
# Loss                                                                        #
# --------------------------------------------------------------------------- #

class KinematicsNetLoss(nn.Module):
    """Joint-training loss combining MSE on each sub-output plus a forward-kinematics joint loss."""

    def __init__(self, body_model: art.ParametricModel, device: torch.device,
                 pl_weight: float = 1.0, pa_weight: float = 1.0,
                 vl_weight: float = 1.0, ra_weight: float = 1.0):
        super().__init__()
        self.body_model = body_model
        self.device = device
        self.pl_weight = pl_weight
        self.pa_weight = pa_weight
        self.vl_weight = vl_weight
        self.ra_weight = ra_weight
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, T, 198], target: [B, T, 246]
        pRL_pred, pRJ_pred = pred[:, :, :15], pred[:, :, 15:84]
        vM_pred,  RRJ_pred = pred[:, :, 84:102], pred[:, :, 102:198]

        pRL_target = target[:, :, :15]
        pRJ_target = target[:, :, 15:84]
        vM_target = target[:, :, 84:102]
        RRJ_target = target[:, :, 102:246]

        # Decode RRJ predictions back to a full 24-joint local pose for joint-position loss.
        B, T, _ = RRJ_pred.shape
        ra_mat = art.math.r6d_to_rotation_matrix(RRJ_pred).reshape(-1, 16, 3, 3)
        glb_pose = torch.eye(3, device=self.device).repeat(ra_mat.shape[0], 24, 1, 1)
        glb_pose[:, J_REDUCE[1:]] = ra_mat[:, 1:]
        pose_local = self.body_model.inverse_kinematics_R(glb_pose).view(B, T, 24, 3, 3)
        pose_local[:, :, J_IGNORE] = torch.eye(3, device=self.device)
        pose_local[:, :, 0] = ra_mat[:, 0].reshape(B, T, 3, 3)
        RRJ_pred_flat = pose_local[:, :, J_REDUCE].flatten(2)        # [B, T, 144]

        _, joint_pred = self.body_model.forward_kinematics(
            pose_local.reshape(-1, 24, 3, 3), calc_mesh=False)
        joint_pred = joint_pred.reshape(B, T, 72)[:, :, 3:]          # [B, T, 69]

        return (
            self.pl_weight * self.mse(pRL_pred, pRL_target)
            + self.pa_weight * self.mse(pRJ_pred, pRJ_target)
            + self.vl_weight * self.mse(vM_pred, vM_target)
            + self.ra_weight * self.mse(RRJ_pred_flat, RRJ_target)
            + self.ra_weight * self.mse(joint_pred, pRJ_target)
        )


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #

class KinematicsNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, sequence_length: int, overlap: int,
                 n_imus: int, insole: bool, body_model: art.ParametricModel,
                 device: torch.device):
        self.raw = torch.load(data_path, map_location='cpu', weights_only=True)
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.n_imus = n_imus
        self.insole = insole
        self.body_model = body_model
        self.device = device
        self.sequences: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self._prepare_sequences()

    def _prepare_sequences(self) -> None:
        stride = self.sequence_length - self.overlap
        keys = ('aS', 'wS', 'vS', 'RIS', 'RIM', 'RSB', 'insole', 'pose')
        num_seqs = len(self.raw['pose'])
        for seq_idx in tqdm.tqdm(range(num_seqs), desc='Preparing sequences'):
            seq = {k: self.raw[k][seq_idx] for k in keys}

            RMB, aM, wM, vM = transform_to_world(
                seq['aS'], seq['wS'], seq['vS'], seq['RIS'], seq['RIM'], seq['RSB'])
            RMB, aM, wM = mask_unused_imus(RMB, aM, wM, self.n_imus)

            input_data = build_input_features(aM, wM, RMB, seq['insole'], insole=self.insole)
            label_data, _ = build_label_targets(seq['pose'], vM, self.body_model, self.device)

            seq_len = seq['pose'].shape[0]
            for start in range(0, seq_len - self.sequence_length + 1, stride):
                end = start + self.sequence_length
                self.sequences.append(input_data[start:end])
                self.labels.append(label_data[start:end])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx].to(self.device), self.labels[idx].to(self.device)


def _build_dataloaders(data_path, sequence_length, overlap, n_imus, insole,
                       batch_size, body_model, device):
    dataset = KinematicsNetDataset(data_path, sequence_length, overlap,
                                   n_imus, insole, body_model, device)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Independent training                                                        #
# --------------------------------------------------------------------------- #

def _independent_loss(network_name: str, output: torch.Tensor, target: torch.Tensor,
                      body_model: art.ParametricModel, device: torch.device,
                      mse: nn.Module) -> torch.Tensor:
    """Loss used while training a single sub-network in isolation."""
    if network_name == 'PL':
        return mse(output[:, :, :15], target[:, :, :15])
    if network_name == 'PA':
        return mse(output[:, :, 15:84], target[:, :, 15:84])
    if network_name == 'VL':
        return mse(output[:, :, 84:102], target[:, :, 84:102])
    if network_name == 'RA':
        # Decode RA's 6D prediction, then supervise on both the rotation and joint position.
        B, T = output.shape[:2]
        ra_pred = output[:, :, 102:198]
        ra_mat = art.math.r6d_to_rotation_matrix(ra_pred).reshape(-1, 16, 3, 3)
        glb_pose = torch.eye(3, device=device).repeat(ra_mat.shape[0], 24, 1, 1)
        glb_pose[:, J_REDUCE[1:]] = ra_mat[:, 1:]
        pose_local = body_model.inverse_kinematics_R(glb_pose).view(B, T, 24, 3, 3)
        pose_local[:, :, J_IGNORE] = torch.eye(3, device=device)
        pose_local[:, :, 0] = ra_mat[:, 0].reshape(B, T, 3, 3)
        pose_loss = mse(pose_local[:, :, J_REDUCE].flatten(2), target[:, :, 102:246])

        _, joint_pred = body_model.forward_kinematics(
            pose_local.reshape(-1, 24, 3, 3), calc_mesh=False)
        joint_pred = joint_pred.reshape(B, T, 72)[:, :, 3:]
        joint_loss = mse(joint_pred, target[:, :, 15:84])
        return pose_loss + joint_loss
    raise ValueError(f'Unknown network: {network_name}')


def train_network_independently(model: KinematicsNet, network_name: str,
                                dataloader, num_epochs: int, learning_rate: float,
                                save_dir: str, body_model: art.ParametricModel,
                                device: torch.device, use_wandb: bool) -> None:
    print(f"Starting independent training: {network_name}")

    # Freeze everything, then unfreeze just this sub-network.
    for param in model.parameters():
        param.requires_grad = False
    submodule = _submodule(model, network_name)
    for param in submodule.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(submodule.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for data, target in tqdm.tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            output = model(data, target[:, 0], net=network_name)
            loss = _independent_loss(network_name, output, target, body_model, device, mse)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.6f}')
        if use_wandb:
            log_metrics(use_wandb, {
                f"{network_name}_loss": avg_loss,
                f"{network_name}_epoch": epoch + 1,
            })

    # Persist this sub-network's weights.
    models_dir = os.path.join(save_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    weight_path = os.path.join(models_dir, f'{network_name}.pt')
    torch.save(submodule.state_dict(), weight_path)
    print(f"Independent training completed: {network_name}")
    print(f"Saved {network_name} module weights: {weight_path}")


def load_previous_models(model: KinematicsNet, network: str, models_dir: str,
                         device: torch.device) -> list[str]:
    """Load every other sub-network's weights from disk and verify dependencies are present."""
    others = [n for n in SUBMODULE_ATTR if n != network]
    missing: list[str] = []

    for module_name in others:
        path = os.path.join(models_dir, f'{module_name}.pt')
        if not os.path.exists(path):
            missing.append(module_name)
            continue
        try:
            _submodule(model, module_name).load_state_dict(
                torch.load(path, map_location=device, weights_only=True))
            print(f"Loaded {module_name} module weights: {path}")
        except Exception:
            print(f"Failed to load {module_name} module weights: {path}")
            missing.append(module_name)

    blocking = [m for m in missing if m in DEPENDENCIES[network]]
    if blocking:
        raise FileNotFoundError(
            f"Required pretrained models not found for {network} module training: {blocking}\n"
            f"Training order: PL → PA → VL → RA\n"
            f"Required files: {[os.path.join(models_dir, f'{m}.pt') for m in blocking]}"
        )
    return missing


def train_independent(data_path: str, save_dir: str, networks: list[str],
                      device: torch.device, body_model: art.ParametricModel,
                      n_imus: int, insole: bool, use_wandb: bool,
                      project_name: str, run_name: str) -> None:
    print("=== Executing Independent Training Only ===")
    hp = INDEPENDENT_HPARAMS

    wandb_enabled = init_wandb(
        use_wandb, project_name, run_name,
        config={**hp, 'mode': 'independent', 'n_imus': n_imus, 'insole': insole},
    )

    train_loader, _ = _build_dataloaders(
        data_path, hp['sequence_length'], hp['overlap'], n_imus, insole,
        hp['batch_size'], body_model, device,
    )

    models_dir = os.path.join(save_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    model = KinematicsNet().to(device)

    missing_modules: list[str] = []
    for network in networks:
        missing_modules = load_previous_models(model, network, models_dir, device)
        train_network_independently(
            model, network, train_loader,
            num_epochs=hp['num_epochs'], learning_rate=hp['learning_rate'],
            save_dir=save_dir, body_model=body_model, device=device,
            use_wandb=wandb_enabled,
        )

    if not missing_modules:
        weights_path = os.path.join(models_dir, 'best_model.pt')
        torch.save(model.state_dict(), weights_path)
        print("Independent training completed! Saved entire model weights:", weights_path)
    else:
        print("Independent training completed! "
              "Did not save entire model weights due to missing trained modules.")

    if wandb_enabled:
        wandb.finish()


# --------------------------------------------------------------------------- #
# Joint training                                                              #
# --------------------------------------------------------------------------- #

def _try_load_subnetwork_weights(model: KinematicsNet, networks: list[str],
                                 models_dir: str, device: torch.device) -> bool:
    """Load any per-network checkpoints that exist; return True if at least one was loaded."""
    loaded_any = False
    for module_name in networks:
        path = os.path.join(models_dir, f'{module_name}.pt')
        if os.path.exists(path):
            _submodule(model, module_name).load_state_dict(
                torch.load(path, map_location=device, weights_only=True))
            print(f"Loaded {module_name} module weights: {path}")
            loaded_any = True
    return loaded_any


def train_joint(data_path: str, save_dir: str, networks: list[str],
                device: torch.device, body_model: art.ParametricModel,
                n_imus: int, insole: bool, use_wandb: bool,
                project_name: str, run_name: str) -> None:
    print("=== Executing Joint Training Only ===")
    hp = JOINT_HPARAMS

    wandb_enabled = init_wandb(
        use_wandb, project_name, run_name,
        config={**hp, **JOINT_LOSS_WEIGHTS,
                'mode': 'joint', 'n_imus': n_imus, 'insole': insole},
    )

    train_loader, val_loader = _build_dataloaders(
        data_path, hp['sequence_length'], hp['overlap'], n_imus, insole,
        hp['batch_size'], body_model, device,
    )

    models_dir = os.path.join(save_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    model = KinematicsNet().to(device)
    if not _try_load_subnetwork_weights(model, networks, models_dir, device):
        print("Executing joint training without loading independently trained weights.")

    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
    loss_fn = KinematicsNetLoss(body_model, device, **JOINT_LOSS_WEIGHTS).to(device)

    best_val_loss = float('inf')
    for epoch in range(hp['num_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        for data, target in tqdm.tqdm(train_loader,
                                      desc=f'Training Epoch {epoch+1}/{hp["num_epochs"]}'):
            optimizer.zero_grad()
            output = model(data, target[:, 0], net='all')
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in tqdm.tqdm(val_loader,
                                          desc=f'Validation Epoch {epoch+1}/{hp["num_epochs"]}'):
                output = model(data, target[:, 0], net='all')
                val_loss += loss_fn(output, target).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        if wandb_enabled:
            log_metrics(wandb_enabled, {
                "joint_train_loss": avg_train_loss,
                "joint_val_loss": avg_val_loss,
                "joint_epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pt'))
            print(f"Saved new best model: {avg_val_loss:.6f}")

    print("Joint training completed!")
    if wandb_enabled:
        wandb.finish()


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

SAVE_DIR = 'output/kinematics_net'


def _set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRIP KinematicsNet Training')
    parser.add_argument('--use_wandb', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--project_name', type=str, default='GRIP_KinematicsNet',
                        help='wandb project name')
    parser.add_argument('--dataset', type=str, default='PRISM', choices=['PRISM'])
    parser.add_argument('--mode', type=str, default='independent', choices=['independent', 'joint'],
                        help='training mode')
    parser.add_argument('--n_imus', type=int, default=4, choices=[2, 3, 4, 5, 6],
                        help='number of IMUs')
    parser.add_argument('--insole', type=lambda x: x.lower() == 'true', default=True,
                        help='use insole data')
    parser.add_argument('--fps', type=int, default=100, help='frame rate')
    parser.add_argument('--networks', type=str, nargs='+', default=NETWORK_ORDER,
                        choices=NETWORK_ORDER,
                        help='networks to train (multiple can be specified)')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device index to use (e.g., 0, 1, 2)')
    args = parser.parse_args()

    device = select_device(args.cuda_device)
    print(f"Using device: {device}")
    body_model = load_body_model(device)

    save_dir = SAVE_DIR
    data_path = 'data/preprocessed/kinematics_net/dataset_train.pt'
    print(f'insole: {args.insole}')
    print(f"Save directory: {save_dir}")

    _set_seed()
    run_name = f'{args.dataset}_{args.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    common_kwargs = dict(
        data_path=data_path, save_dir=save_dir,
        device=device, body_model=body_model,
        n_imus=args.n_imus, insole=args.insole,
        use_wandb=args.use_wandb, project_name=args.project_name, run_name=run_name,
    )
    if args.mode == 'independent':
        train_independent(networks=args.networks, **common_kwargs)
    else:
        train_joint(networks=NETWORK_ORDER, **common_kwargs)

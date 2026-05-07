"""KinematicsNet: a four-stage RNN that predicts pose from IMU + insole signals.

Pipeline:
    PL  →  leaf-joint positions
    PA  →  full joint positions   (conditioned on PL outputs)
    VL  →  IMU linear velocity    (conditioned on PA outputs)
    RA  →  joint rotations        (conditioned on PA outputs)
"""
__all__ = ['KinematicsNet']

import torch
import torch.nn as nn
from torch.nn.functional import relu


# Output dimensions for each sub-network.
PRL_DIM = 15      # leaf joints (5 joints * 3)
PRJ_DIM = 69      # body joints (23 joints * 3)
VM_DIM = 18       # IMU velocity (6 IMUs * 3)
RRJ_DIM = 96      # joint rotations as 6D (16 joints * 6)

# Input feature dim: aM(18) + wM(18) + RMB(54) + iS(10) = 100
FEATURE_DIM = 100
# RA's init vector encodes 16 joint rotation matrices (16 * 9).
RRJ_INIT_DIM = 144

_HIDDEN_SIZE = 512
_NUM_RNN_LAYERS = 3
_DROPOUT = 0.4


class RNN(nn.Module):
    """LSTM with optional input projection. Input is ``[B, T, *]``."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int,
                 num_rnn_layer: int, rnn_type: str = 'lstm',
                 bidirectional: bool = False, input_linear: bool = True,
                 dropout: float = 0.):
        super().__init__()
        rnn_input = hidden_size if input_linear else input_size
        self.rnn = getattr(nn, rnn_type.upper())(
            rnn_input, hidden_size, num_rnn_layer,
            bidirectional=bidirectional, dropout=dropout,
        )
        self.input_proj = nn.Linear(input_size, hidden_size) if input_linear else nn.Identity()
        self.output_proj = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.input_linear = input_linear

    def forward(self, x, init=None):
        # [B, T, *] -> [T, B, *] for the LSTM API
        x = x.permute(1, 0, 2)
        if self.input_linear:
            x = self.dropout(relu(self.input_proj(x)))
        x = self.rnn(x, init)[0]
        x = self.output_proj(x)
        return [x[:, i] for i in range(x.shape[1])]


class RNNWithInit(RNN):
    """LSTM whose hidden / cell state is predicted from a per-sequence init vector."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int,
                 num_rnn_layer: int, init_size: int = None,
                 bidirectional: bool = False, input_linear: bool = True,
                 dropout: float = 0., layer_norm: bool = False,
                 rnn_type: str = 'lstm'):
        assert rnn_type.upper() in ('LSTM', 'LNLSTM') and bidirectional is False
        super().__init__(input_size, output_size, hidden_size, num_rnn_layer,
                         rnn_type, bidirectional, input_linear, dropout)
        self.num_layers = num_rnn_layer
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        directions = 2 if bidirectional else 1
        out_size = 2 * directions * num_rnn_layer * hidden_size
        self.init_net = nn.Sequential(
            nn.Linear(init_size or output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * num_rnn_layer),
            nn.ReLU(),
            nn.Linear(hidden_size * num_rnn_layer, out_size),
            nn.LayerNorm(out_size) if layer_norm else nn.Identity(),
        )

    def forward(self, x, x_init):
        nd = self.num_layers * (2 if self.bidirectional else 1)
        nh = self.hidden_size
        h, c = self.init_net(x_init).view(-1, 2, nd, nh).permute(1, 2, 0, 3).contiguous()
        return super().forward(x, (h, c))


def _make_subnet(input_size: int, output_size: int, init_size: int = None) -> RNNWithInit:
    """Build the canonical RNNWithInit configuration shared by every sub-network."""
    return RNNWithInit(
        input_size=input_size,
        output_size=output_size,
        init_size=init_size,
        hidden_size=_HIDDEN_SIZE,
        num_rnn_layer=_NUM_RNN_LAYERS,
        dropout=_DROPOUT,
        input_linear=False,
    )


class KinematicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Each sub-network conditions on the raw features plus upstream outputs.
        self.plnet = _make_subnet(input_size=FEATURE_DIM,             output_size=PRL_DIM)
        self.panet = _make_subnet(input_size=FEATURE_DIM + PRL_DIM,   output_size=PRJ_DIM)
        self.vlnet = _make_subnet(input_size=FEATURE_DIM + PRJ_DIM,   output_size=VM_DIM)
        self.ranet = _make_subnet(input_size=FEATURE_DIM + PRJ_DIM,   output_size=RRJ_DIM,
                                  init_size=RRJ_INIT_DIM)

    def forward(self, x: torch.Tensor, y: torch.Tensor, net: str = 'all') -> torch.Tensor:
        """Run the four sub-networks.

        Args:
            x: [B, T, 100]  features (aM | wM | RMB | iS).
            y: [B, 246]     per-sequence init vector laid out as
                              [:15]      pRL  init   (leaf joints, 5*3)
                              [15:84]    pRJ  init   (joints, 23*3)
                              [84:102]   vM   init   (6 IMU velocities)
                              [102:246]  RRJ  init   (16 joint rotation matrices)
            net: which branch's output should be populated.
                 ``'PL'``, ``'PA'``, ``'VL'``, ``'RA'`` enable the matching branch
                 for independent training; ``'all'`` runs the full chain.

        Returns:
            [B, T, 198] = pRL(15) | pRJ(69) | vM(18) | RRJ(96)
        """
        B, T, _ = x.shape
        device = x.device

        # Slice indices into the init vector y.
        i_pRL = (0, PRL_DIM)
        i_pRJ = (PRL_DIM, PRL_DIM + PRJ_DIM)
        i_vM = (i_pRJ[1], i_pRJ[1] + VM_DIM)
        i_RRJ = (i_vM[1], i_vM[1] + RRJ_INIT_DIM)

        def run(subnet, inputs, init_slice):
            init = y[:, init_slice[0]:init_slice[1]]
            return torch.stack(subnet.forward(inputs, init), dim=0)

        # PL — always runs.
        pRL = run(self.plnet, x, i_pRL)

        # PA — runs unless we are training RA in isolation.
        if net in ('PA', 'VL', 'all'):
            pRJ = run(self.panet, torch.cat((x, pRL), dim=2), i_pRJ)
        else:
            pRJ = torch.zeros(B, T, PRJ_DIM, device=device)

        # VL — runs only when explicitly requested or for full forward.
        if net in ('VL', 'all'):
            vM = run(self.vlnet, torch.cat((x, pRJ), dim=2), i_vM)
        else:
            vM = torch.zeros(B, T, VM_DIM, device=device)

        # RA — runs only when explicitly requested or for full forward.
        if net in ('RA', 'all'):
            RRJ = run(self.ranet, torch.cat((x, pRJ), dim=2), i_RRJ)
        else:
            RRJ = torch.zeros(B, T, RRJ_DIM, device=device)

        return torch.cat((pRL, pRJ, vM, RRJ), dim=-1)

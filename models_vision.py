import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


class Encoder(nn.Module):
    """
    Conv1d encoder for 1D RGB vision strips.
    Accepts flattened obs (B, 480) or image-like (B,1,160,3)/(1,160,3).
    Internally reshapes to (B, C=3, L=160).
    """
    def __init__(self, input_dim: int, latent_dim: int, aux_dim: int = 0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.aux_dim = int(aux_dim)
        if self.input_dim % 3 != 0:
            raise ValueError(f"Encoder expects input_dim divisible by 3 for RGB strip, got {self.input_dim}")
        self.seq_len = self.input_dim // 3

        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=2, padding=2),  # 160 -> 80
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 80 -> 40
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # 40 -> 20
            nn.ReLU(),
        )
        self.conv_out_len = (self.seq_len + 7) // 8  # valid for seq_len divisible by 8 (160 -> 20)
        self.fc = nn.Sequential(
            nn.Linear(128 * self.conv_out_len + self.aux_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.dim() == 2:
            # (B, 480) -> (B, 160, 3)
            x = x.reshape(x.shape[0], self.seq_len, 3)
        elif x.dim() == 3:
            # Could be (1,160,3) single obs or (B,160,3)
            if x.shape[-1] == 3:
                if x.shape[0] == 1 and x.shape[1] == self.seq_len and x.shape[2] == 3:
                    x = x.unsqueeze(0)  # treat as single sample => (B=1,1,160,3)
                else:
                    x = x  # (B,160,3)
            else:
                raise ValueError(f"Unsupported 3D input shape for Encoder: {tuple(x.shape)}")
        if x.dim() == 4:
            # Expect (B,1,160,3)
            if x.shape[1] == 1 and x.shape[2] == self.seq_len and x.shape[3] == 3:
                x = x[:, 0, :, :]  # -> (B,160,3)
            else:
                raise ValueError(f"Unsupported 4D input shape for Encoder: {tuple(x.shape)}")

        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Encoder expected (...,{self.seq_len},3), got {tuple(x.shape)}")

        # (B,L,C) -> (B,C,L)
        return x.permute(0, 2, 1).contiguous()

    def forward(self, vision, aux_obs=None):
        # Accept vision as flattened (B, 3*L) or image-like (B,1,L,3)/(1,L,3).
        x = self._to_bcl(vision)  # (B,3,L)
        h = self.conv(x)
        h = h.flatten(start_dim=1)  # (B, 128*L')
        if self.aux_dim > 0:
            if aux_obs is None:
                r = h.new_zeros((h.shape[0], self.aux_dim))
            else:
                r = aux_obs
                if r.dim() == 1:
                    r = r.unsqueeze(-1)
                r = r.reshape(r.shape[0], -1).to(h.dtype)
                if r.shape[1] != self.aux_dim:
                    if self.aux_dim == 1 and r.numel() == h.shape[0]:
                        r = r.view(h.shape[0], 1)
                    else:
                        raise ValueError(f"aux_obs has shape {tuple(r.shape)} but expected (*,{self.aux_dim})")
            h = torch.cat([h, r], dim=-1)
        h = self.fc(h)
        mu = self.mu_head(h)
        logvar = torch.clamp(self.logvar_head(h), min=-10.0, max=10.0)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder for continuous vision strip (Gaussian/L1-style reconstruction)
    plus auxiliary head for hunger.
    """
    def __init__(self, latent_dim, output_dim, dropout_p=0.1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.latent_dim = int(latent_dim)
        self.dropout_p = float(dropout_p)
        if self.output_dim % 3 != 0:
            raise ValueError(f"Decoder expects output_dim divisible by 3 for RGB strip, got {self.output_dim}")
        self.seq_len = self.output_dim // 3
        if self.seq_len % 8 != 0:
            raise ValueError(f"Decoder expects seq_len divisible by 8 for current ConvTranspose stack, got {self.seq_len}")

        self.base_len = self.seq_len // 8  # 160 -> 20
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(256, 128 * self.base_len),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.ConvTranspose1d(32, 3, kernel_size=4, stride=2, padding=1),
        )
        self.deconv_logvar = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.ConvTranspose1d(32, 3, kernel_size=4, stride=2, padding=1),
        )
        self.hunger_head = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 1),
        )

    @contextmanager
    def mc_dropout(self):
        was_training = self.training
        try:
            self.train()
            yield
        finally:
            if not was_training:
                self.eval()

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h = self.fc(z)
        h = h.view(z.shape[0], 128, self.base_len)
        x_mean_bcl = self.deconv(h)
        x_logvar_bcl = self.deconv_logvar(h)
        x_mean_blc = x_mean_bcl.permute(0, 2, 1)
        x_logvar_blc = x_logvar_bcl.permute(0, 2, 1)
        vision_mean = torch.sigmoid(x_mean_blc.reshape(z.shape[0], -1))
        vision_logvar = torch.clamp(x_logvar_blc.reshape(z.shape[0], -1), min=-4.0, max=1.0)
        hunger_pred = torch.sigmoid(self.hunger_head(z))
        return {"vision_mean": vision_mean, "vision_logvar": vision_logvar, "hunger_pred": hunger_pred}


class TransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim, dropout_p=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.dropout_p = dropout_p
        self.fc1 = nn.Linear(latent_dim + action_dim, 256)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

    @contextmanager
    def mc_dropout(self):
        was_training = self.training
        try:
            self.train()
            yield
        finally:
            if not was_training:
                self.eval()

    def forward(self, z_prev, action_onehot, enable_dropout=False):
        x = torch.cat([z_prev, action_onehot], dim=1)
        h = F.relu(self.fc1(x))
        if enable_dropout or self.training:
            h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        if enable_dropout or self.training:
            h = self.dropout2(h)
        h = F.relu(self.fc3(h))
        if enable_dropout or self.training:
            h = self.dropout3(h)
        mu = self.mu_head(h)
        logvar = torch.clamp(self.logvar_head(h), min=-10.0, max=10.0)
        return mu, logvar


class HabitNetwork(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, z):
        return self.fc_net(z)

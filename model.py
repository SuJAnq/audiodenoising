# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW


def crop_to_match(encoder_feat: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
    """Center-crop or pad encoder feature to match decoder spatial size for skip connections."""
    diff_y = encoder_feat.size(2) - decoder_feat.size(2)
    diff_x = encoder_feat.size(3) - decoder_feat.size(3)
    # Crop if encoder is larger, pad if smaller
    if diff_y >= 0 and diff_x >= 0:
        return encoder_feat[:, :, diff_y // 2 : encoder_feat.size(2) - (diff_y - diff_y // 2), diff_x // 2 : encoder_feat.size(3) - (diff_x - diff_x // 2)]
    # Pad order: (left, right, top, bottom)
    pad = [max(-diff_x // 2, 0), max(-diff_x - (-diff_x // 2), 0), max(-diff_y // 2, 0), max(-diff_y - (-diff_y // 2), 0)]
    return F.pad(encoder_feat, pad)

# Double Convolution Block from the original U-Net (Ronneberger et al. 2015)
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Encoder(nn.Module):
    """Four-level encoder with max-pooling, matching spectrogram dimensions (513 x T)."""
    def __init__(self, in_ch: int = 1, base_ch: int = 64, dropout: float = 0.0):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch, dropout=dropout)
        self.enc2 = DoubleConv(base_ch, base_ch * 2, dropout=dropout)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4, dropout=dropout)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8, dropout=dropout)
        self.center = DoubleConv(base_ch * 8, base_ch * 16, dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.center(self.pool(x4))
        return x1, x2, x3, x4, x5

class Decoder(nn.Module):
    def __init__(self, out_ch: int = 1, base_ch: int = 64, dropout: float = 0.0, final_activation: str = "softplus"):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch, dropout=dropout)

        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)
        self.final_activation = final_activation

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor) -> torch.Tensor:
        d4 = self.up4(x5)
        d4 = self.dec4(torch.cat([d4, crop_to_match(x4, d4)], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, crop_to_match(x3, d3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, crop_to_match(x2, d2)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, crop_to_match(x1, d1)], dim=1))

        out = self.final(d1)
        if self.final_activation == "softplus":
            return F.softplus(out)
        if self.final_activation == "relu":
            return F.relu(out)
        return out

class UNet(nn.Module):
    """Standard U-Net for magnitude spectrogram denoising (Ronneberger et al. 2015).

    When ``mask_mode=True`` the network predicts a [0, 1] ratio mask (via sigmoid)
    that is element-wise multiplied with the input spectrogram.  This guarantees
    the output can reach exactly zero (full suppression) and never exceed the
    input energy — eliminating the residual hiss that softplus cannot remove.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 64,
                 dropout: float = 0.0, final_activation: str = "softplus",
                 mask_mode: bool = False):
        super().__init__()
        self.mask_mode = mask_mode
        self.encoder = Encoder(in_ch=in_ch, base_ch=base_ch, dropout=dropout)
        # In mask mode, the final activation is handled manually (sigmoid)
        dec_activation = "none" if mask_mode else final_activation
        self.decoder = Decoder(out_ch=out_ch, base_ch=base_ch, dropout=dropout, final_activation=dec_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad input to be divisible by 16 (2^4) because of 4 pooling layers
        h, w = x.shape[2], x.shape[3]
        pad_h = (16 - (h % 16)) % 16
        pad_w = (16 - (w % 16)) % 16
        
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_padded = x

        x1, x2, x3, x4, x5 = self.encoder(x_padded)
        out = self.decoder(x1, x2, x3, x4, x5)

        # Crop back to original size if padded
        if pad_h > 0 or pad_w > 0:
            out = out[..., :h, :w]

        if self.mask_mode:
            # Predict a ratio mask in [0, 1] and apply it to the input
            mask = torch.sigmoid(out)
            return mask * x
            
        return out


def stft_transform(waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute magnitude and phase; waveform shape (T,)."""
    stft = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW.to(waveform.device),
        return_complex=True,
    )
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    return mag, phase

# Kaiming (He) Initialization for UNet weights
def init_kaiming(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)



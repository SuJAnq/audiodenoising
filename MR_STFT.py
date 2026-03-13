import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConvergenceLoss(nn.Module):
    """Spectral Convergence Loss."""
    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        """
        x_mag: Magnitude spectrogram of predicted waveform
        y_mag: Magnitude spectrogram of target waveform
        """
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-7)


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT Magnitude Loss."""
    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        """
        x_mag: Magnitude spectrogram of predicted waveform
        y_mag: Magnitude spectrogram of target waveform
        """
        return F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))


class STFTLoss(nn.Module):
    """
    Computes the STFT loss for a single resolution.
    Consists of Spectral Convergence Loss (SC) and Log STFT Magnitude Loss (Mag).
    """
    def __init__(self, fft_size=1024, hop_size=120, win_length=600, window="hann_window"):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        # Register window as buffer to be moved to GPU automatically
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.sc_loss = SpectralConvergenceLoss()
        self.mag_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """
        x: Predicted waveform (B, T) or (B, 1, T)
        y: Target waveform (B, T) or (B, 1, T)
        """
        # Ensure (B, T)
        if x.dim() == 3: x = x.squeeze(1)
        if y.dim() == 3: y = y.squeeze(1)

        # Compute STFT
        # center=True is default and important for alignment
        x_stft = torch.stft(x, n_fft=self.fft_size, hop_length=self.hop_size, 
                            win_length=self.win_length, window=self.window, 
                            return_complex=True)
        y_stft = torch.stft(y, n_fft=self.fft_size, hop_length=self.hop_size, 
                            win_length=self.win_length, window=self.window, 
                            return_complex=True)

        # Magnitudes
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        # Spectral Convergence Loss
        sc_l = self.sc_loss(x_mag, y_mag)

        # Log STFT Magnitude Loss
        mag_l = self.mag_loss(x_mag, y_mag)

        return sc_l, mag_l


class MultiResolutionSTFTLoss(nn.Module):
    """
    Compute STFT loss at multiple resolutions to capture both time and frequency structures.
    """
    def __init__(self,
                 fft_sizes=[512, 1024, 2048],
                 hop_sizes=[50, 120, 240],
                 win_lengths=[240, 600, 1200],
                 window="hann_window",
                 factor_sc=0.1,
                 factor_mag=0.1):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, ss, wl, window))
        
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc, mag = f(x, y)
            sc_loss += sc
            mag_loss += mag
        
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss + self.factor_mag * mag_loss


class CombinedLoss(nn.Module):
    """
    Combines Waveform MSE Loss and Multi-Resolution STFT Loss.
    L_total = lambda1 * MSE + lambda2 * MRSTFT
    """
    def __init__(self, 
                 lambda1=100.0, 
                 lambda2=1.0,
                 fft_sizes=[512, 1024, 2048],
                 hop_sizes=[50, 120, 240],
                 win_lengths=[240, 600, 1200],
                 window="hann_window"):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mse_loss = nn.MSELoss()
        self.mrstft_loss = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths, window)

    def forward(self, pred_wav, target_wav):
        """
        pred_wav: Predicted waveform (B, 1, T) or (B, T)
        target_wav: Target waveform (B, 1, T) or (B, T)
        """
        mse = self.mse_loss(pred_wav, target_wav)
        mrstft = self.mrstft_loss(pred_wav, target_wav)
        
        return self.lambda1 * mse + self.lambda2 * mrstft, mse, mrstft

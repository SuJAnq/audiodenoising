import torch
import math

from config import get_config

cfg = get_config()


def _apply_light_postfilter(waveform: torch.Tensor, *,
                            sample_rate: int = None,
                            cutoff_hz: float = None,
                            strength: float = None) -> torch.Tensor:
    """Apply a gentle one-pole low-pass mix to reduce high-frequency hiss.

    waveform: (T,) or (B, T)
    strength: 0.0 -> unchanged, 1.0 -> fully low-passed
    """
    local_cfg = cfg
    sample_rate = sample_rate if sample_rate is not None else local_cfg.SAMPLE_RATE
    cutoff_hz = cutoff_hz if cutoff_hz is not None else local_cfg.POSTFILTER_CUTOFF_HZ
    strength = strength if strength is not None else local_cfg.POSTFILTER_STRENGTH

    if strength <= 0.0:
        return waveform

    cutoff_hz = max(1.0, min(float(cutoff_hz), (sample_rate * 0.5) - 1.0))
    strength = max(0.0, min(float(strength), 1.0))

    alpha = math.exp(-2.0 * math.pi * cutoff_hz / float(sample_rate))

    single = waveform.dim() == 1
    if single:
        waveform = waveform.unsqueeze(0)

    filtered = torch.empty_like(waveform)
    filtered[:, 0] = waveform[:, 0]
    for t in range(1, waveform.size(1)):
        filtered[:, t] = (1.0 - alpha) * waveform[:, t] + alpha * filtered[:, t - 1]

    out = (1.0 - strength) * waveform + strength * filtered
    return out.squeeze(0) if single else out


def _restore_nyquist_bin(mag: torch.Tensor, phase: torch.Tensor, *, n_fft: int, ref_mag: torch.Tensor = None):
    """If spectrogram has 512 bins from a 1024 FFT, append the missing Nyquist row.

    STFT with n_fft=1024 yields 513 bins. The training pipeline crops to 512
    for UNet convenience. Before ISTFT we re-attach the Nyquist bin.
    
    Improved: Instead of zero-padding (which causes artifacts), we copy
    the Nyquist bin from the reference noisy magnitude if available.
    """
    expected_bins = (n_fft // 2) + 1  # e.g., 1024 -> 513
    if mag.dim() == 4:
        # (B,1,F,T)
        freq_dim = 2
    else:
        # (B,F,T) or (F,T)
        freq_dim = 1 if mag.dim() == 3 else 0

    current_bins = mag.size(freq_dim)
    if current_bins == expected_bins:
        return mag, phase
    if current_bins != expected_bins - 1:
        raise ValueError(f"Unexpected freq bins: {current_bins}; expected {expected_bins} or {expected_bins-1}")

    pad_shape = list(mag.shape)
    pad_shape[freq_dim] = 1
    
    # Use reference magnitude for the Nyquist bin if available (preserve high-freq noise is better than hard zero)
    if ref_mag is not None:
        # Ensure ref_mag matches dimensions roughly or slice accordingly
        # ref_mag is likely (B, 1, 513, T) or similar.
        # We need the last bin from it.
        if ref_mag.dim() == mag.dim():
             # Access last bin along freq_dim
             nyquist_bin = torch.narrow(ref_mag, freq_dim, -1, 1)
             pad_mag = nyquist_bin
        else:
             pad_mag = torch.zeros(pad_shape, device=mag.device, dtype=mag.dtype)
    else:
        pad_mag = torch.zeros(pad_shape, device=mag.device, dtype=mag.dtype)

    mag = torch.cat([mag, pad_mag], dim=freq_dim)

    if phase is not None:
        pad_phase = torch.zeros(pad_shape, device=phase.device, dtype=phase.dtype)
        # If we copied magnitude, we should ideally copy phase too, but phase is passed separately usually.
        # For keeping it simple, 0 phase for Nyquist is acceptable provided magnitude isn't hard-cut zero.
        # (Ideally we'd accept ref_phase too, but ref_mag is the main perceptual fix).
        phase = torch.cat([phase, pad_phase], dim=freq_dim)

    return mag, phase


def reconstruct_waveform_from_mag_and_phase(mag: torch.Tensor, phase: torch.Tensor, *,
                                            n_fft: int = None,
                                            hop_length: int = None,
                                            win_length: int = None,
                                            window: torch.Tensor = None,
                                            use_log_mag: bool = None,
                                            ref_mag: torch.Tensor = None,
                                            ) -> torch.Tensor:
    """Reconstruct time waveform from magnitude and phase tensors.

    mag: (B, 1, F, T) or (F, T) or (B, F, T)
    phase: (B, F, T) or (F, T)
    ref_mag: Optional (B, 1, F_full, T) original noisy magnitude to borrow Nyquist bin from.

    Returns float32 waveform tensor (B, T_samples) if batch provided, else 1D tensor.
    """
    single = False
    if mag.dim() == 3 and mag.size(0) != 1 and mag.size(1) != 1:
        # mag shape (B, F, T)
        pass
    # normalize shapes to (B, F, T)
    if mag.dim() == 4:
        # (B, 1, F, T)
        mag = mag.squeeze(1)
    if mag.dim() == 2:
        mag = mag.unsqueeze(0)
        single = True

    if phase is None:
        raise ValueError("phase must be provided for noisy-phase reconstruction")
    if phase.dim() == 2:
        phase = phase.unsqueeze(0)

    # undo log1p if used during preprocessing/modeling
    local_cfg = cfg
    n_fft = n_fft if n_fft is not None else local_cfg.N_FFT
    hop_length = hop_length if hop_length is not None else local_cfg.HOP_LENGTH
    win_length = win_length if win_length is not None else local_cfg.WIN_LENGTH
    use_log_mag = use_log_mag if use_log_mag is not None else local_cfg.USE_LOG_MAG

    if use_log_mag:
        mag = torch.expm1(mag)
        
    # If ref_mag was provided in log1p domain (and use_log_mag is True), 
    # we must linearize it to match the 'mag' variable we just linearized above.
    # Note: ref_mag is optional, so check existence first.
    if ref_mag is not None and use_log_mag and use_log_mag is True:
        ref_mag_lin = torch.expm1(ref_mag)
    else:
        ref_mag_lin = ref_mag

    # re-attach missing Nyquist bin if model used 512 bins
    mag, phase = _restore_nyquist_bin(mag, phase, n_fft=n_fft, ref_mag=ref_mag_lin)

    # build complex STFT via polar form
    # mag and phase should be real tensors of same shape (B, F, T)
    complex_spec = torch.polar(mag, phase)

    # prepare window
    if window is None:
        window = torch.hann_window(win_length).to(complex_spec.device)

    # perform inverse STFT per sample
    waveforms = []
    for i in range(complex_spec.size(0)):
        spec = complex_spec[i].to(torch.complex64)
        waveform = torch.istft(spec,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               win_length=win_length,
                               window=window,
                               length=None)
        waveforms.append(waveform)

    if single:
        return waveforms[0]
    return torch.stack(waveforms, dim=0)


def griffin_lim_reconstruct(mag: torch.Tensor, *,
                            n_iter: int = None,
                            n_fft: int = None,
                            hop_length: int = None,
                            win_length: int = None,
                            use_log_mag: bool = None,
                            ) -> torch.Tensor:
    """Reconstruct a waveform from a magnitude spectrogram using the Griffin-Lim algorithm.

    mag: (B, 1, F, T) or (B, F, T) or (F, T)
    Returns float32 waveform tensor (B, T_samples), or 1D tensor if single input.

    Parameters are read from config if not explicitly provided.
    """
    local_cfg = cfg
    n_iter      = n_iter      if n_iter      is not None else local_cfg.GRIFFIN_LIM_ITER
    n_fft       = n_fft       if n_fft       is not None else local_cfg.N_FFT
    hop_length  = hop_length  if hop_length  is not None else local_cfg.HOP_LENGTH
    win_length  = win_length  if win_length  is not None else local_cfg.WIN_LENGTH
    use_log_mag = use_log_mag if use_log_mag is not None else local_cfg.USE_LOG_MAG

    # Normalise shape to (B, F, T)
    single = False
    if mag.dim() == 4:
        mag = mag.squeeze(1)          # (B, 1, F, T) -> (B, F, T)
    if mag.dim() == 2:
        mag = mag.unsqueeze(0)        # (F, T) -> (1, F, T)
        single = True

    # Undo log1p if the spectrogram was stored in log-magnitude
    if use_log_mag:
        mag = torch.expm1(mag)

    # Re-attach Nyquist bin if model produced (N_FFT//2) bins instead of (N_FFT//2 + 1)
    dummy_phase = torch.zeros_like(mag)
    mag, _ = _restore_nyquist_bin(mag, dummy_phase, n_fft=n_fft)

    window = torch.hann_window(win_length).to(mag.device)

    waveforms = []
    for b in range(mag.size(0)):
        mag_b = mag[b]  # (F, T)

        # Initialise with random phase
        phase = torch.rand_like(mag_b) * 2 * math.pi - math.pi

        for _ in range(n_iter):
            # Build complex spectrogram from current magnitude + phase estimate
            complex_spec = torch.polar(mag_b, phase).to(torch.complex64)

            # ISTFT -> estimated waveform
            waveform = torch.istft(complex_spec,
                                   n_fft=n_fft,
                                   hop_length=hop_length,
                                   win_length=win_length,
                                   window=window)

            # STFT -> updated phase estimate
            new_complex = torch.stft(waveform,
                                     n_fft=n_fft,
                                     hop_length=hop_length,
                                     win_length=win_length,
                                     window=window,
                                     return_complex=True)
            phase = torch.angle(new_complex)

        waveforms.append(waveform)

    if single:
        return waveforms[0]
    return torch.stack(waveforms, dim=0)


def reconstruct_waveform_auto(mag: torch.Tensor, phase: torch.Tensor = None, *,
                              ref_mag: torch.Tensor = None,
                              prefer_input_phase: bool = None,
                              apply_postfilter: bool = None,
                              postfilter_cutoff_hz: float = None,
                              postfilter_strength: float = None,
                              apply_spectral_gate: bool = None,
                              spectral_gate_threshold: float = None,
                              apply_wiener: bool = None,
                              wiener_beta: float = None) -> torch.Tensor:
    """Auto-select reconstruction method and optionally apply anti-hiss postfilter.

    Policy:
      - If `phase` is available and `prefer_input_phase` is True: ISTFT with input phase.
      - Otherwise: Griffin-Lim reconstruction.

    Post-processing chain (all optional, executed in order):
      1. Spectral gating — zeroes out time-frequency bins below a power threshold
      2. Wiener-style post-filter — attenuates bins with low predicted SNR
      3. Light one-pole low-pass — reduces high-freq hiss in the time domain
    """
    local_cfg = cfg
    prefer_input_phase = prefer_input_phase if prefer_input_phase is not None else local_cfg.PREFER_INPUT_PHASE_RECON
    apply_postfilter = apply_postfilter if apply_postfilter is not None else local_cfg.APPLY_POSTFILTER
    apply_spectral_gate = apply_spectral_gate if apply_spectral_gate is not None else getattr(local_cfg, 'APPLY_SPECTRAL_GATE', True)
    spectral_gate_threshold = spectral_gate_threshold if spectral_gate_threshold is not None else getattr(local_cfg, 'SPECTRAL_GATE_THRESHOLD', 0.02)
    apply_wiener = apply_wiener if apply_wiener is not None else getattr(local_cfg, 'APPLY_WIENER_POSTFILTER', False)
    wiener_beta = wiener_beta if wiener_beta is not None else getattr(local_cfg, 'WIENER_BETA', 0.02)

    if phase is not None and prefer_input_phase:
        waveform = reconstruct_waveform_from_mag_and_phase(mag, phase, ref_mag=ref_mag)
    else:
        waveform = griffin_lim_reconstruct(mag)

    # --- Spectral post-processing ---
    if apply_spectral_gate or apply_wiener:
        waveform = _spectral_post_process(
            waveform,
            sample_rate=local_cfg.SAMPLE_RATE,
            apply_gate=apply_spectral_gate,
            gate_threshold=spectral_gate_threshold,
            apply_wiener=apply_wiener,
            wiener_beta=wiener_beta,
        )

    if apply_postfilter:
        waveform = _apply_light_postfilter(
            waveform,
            sample_rate=local_cfg.SAMPLE_RATE,
            cutoff_hz=postfilter_cutoff_hz,
            strength=postfilter_strength,
        )

    return waveform


# ---------------------------------------------------------------------------
# Spectral post-processing: gate + Wiener filter
# ---------------------------------------------------------------------------

def _spectral_post_process(waveform: torch.Tensor, *,
                           sample_rate: int = 16000,
                           n_fft: int = None,
                           hop_length: int = None,
                           win_length: int = None,
                           apply_gate: bool = True,
                           gate_threshold: float = 0.02,
                           apply_wiener: bool = False,
                           wiener_beta: float = 0.02) -> torch.Tensor:
    """Apply spectral-domain post-processing to remove residual static / rain noise.

    1. **Spectral gate**: Estimate a noise floor from the quietest 10% of frames,
       then zero out any T-F bin whose magnitude falls below ``gate_threshold``
       times that floor.  This removes the faint "rain" pattern without touching
       speech-dominant bins.
    2. **Wiener post-filter**: Compute a soft mask H = |S|^2 / (|S|^2 + beta)
       and multiply the complex spectrum by it.  Beta controls suppression
       strength (higher -> more aggressive).

    Both operate in the STFT domain and reconstruct via ISTFT, preserving length.
    """
    local_cfg = cfg
    n_fft = n_fft or local_cfg.N_FFT
    hop_length = hop_length or local_cfg.HOP_LENGTH
    win_length = win_length or local_cfg.WIN_LENGTH

    single = waveform.dim() == 1
    if single:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(win_length, device=waveform.device)
    results = []

    for b in range(waveform.size(0)):
        sig = waveform[b]
        orig_len = sig.size(0)

        spec = torch.stft(sig, n_fft=n_fft, hop_length=hop_length,
                          win_length=win_length, window=window,
                          return_complex=True)
        mag = spec.abs()
        phase = spec.angle()

        if apply_gate:
            # Estimate noise floor from the quietest 10 % of frames
            frame_energy = mag.pow(2).mean(dim=0)  # (T,)
            k = max(1, int(frame_energy.size(0) * 0.10))
            noise_energy, _ = frame_energy.topk(k, largest=False)
            noise_floor = noise_energy.mean().sqrt()  # RMS of quiet frames

            # Soft gate: suppress bins below threshold * noise_floor
            threshold = gate_threshold * noise_floor
            gate_mask = (mag > threshold).float()
            # Smooth the mask slightly to avoid hard clickiness
            # Use a small 3x3 average pool if possible
            if gate_mask.dim() == 2:
                gm = gate_mask.unsqueeze(0).unsqueeze(0)
                gm = torch.nn.functional.avg_pool2d(gm, kernel_size=3, stride=1, padding=1)
                gate_mask = gm.squeeze(0).squeeze(0)
                gate_mask = (gate_mask > 0.5).float()
            mag = mag * gate_mask

        if apply_wiener:
            power = mag.pow(2)
            wiener_mask = power / (power + wiener_beta)
            mag = mag * wiener_mask

        spec_clean = torch.polar(mag, phase)
        recon = torch.istft(spec_clean, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window,
                            length=orig_len)
        results.append(recon)

    out = torch.stack(results, dim=0)
    return out.squeeze(0) if single else out

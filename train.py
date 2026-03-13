# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

import os
import argparse
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import get_config
from dataset import DenoisingDataset
from utilis import reconstruct_waveform_from_mag_and_phase
from model import UNet, init_kaiming

try:
    from MR_STFT import MultiResolutionSTFTLoss, CombinedLoss
except ImportError:
    # Fallback if MR_STFT.py is missing or broken, though we expect it to be there
    MultiResolutionSTFTLoss = None
    CombinedLoss = None


cfg = get_config()


def setup_logger(log_dir: str) -> logging.Logger:
    """Configure a logger that writes to both console and a timestamped log file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-runs

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


class PlaceholderMRSTFTLoss(nn.Module):
    """Stand-in for Multi-Resolution STFT loss; uses simple L1 on magnitudes."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target)


class HybridLoss(nn.Module):
    """Combines MR-STFT and MSE loss."""
    def __init__(self, mrstft_loss, alpha=0.5):
        super().__init__()
        self.mrstft = mrstft_loss
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred_mag, target_mag, pred_wav, target_wav):
        loss_mrstft = self.mrstft(pred_wav, target_wav)
        loss_mse = self.mse(pred_mag, target_mag)
        # alpha controls MR-STFT contribution
        # If alpha=0.3, then: 0.3 * MRSTFT + 0.7 * MSE
        return self.alpha * loss_mrstft + (1 - self.alpha) * loss_mse


class HybridL1Loss(nn.Module):
    """Combines MR-STFT and L1 loss.
    
    L1 (median-seeking) is more robust to noisy targets than MSE (mean-seeking),
    making this better suited for Noise2Noise training where targets contain noise.
    """
    def __init__(self, mrstft_loss, alpha=0.5):
        super().__init__()
        self.mrstft = mrstft_loss
        self.l1 = nn.L1Loss()
        self.alpha = alpha

    def forward(self, pred_mag, target_mag, pred_wav, target_wav):
        loss_mrstft = self.mrstft(pred_wav, target_wav)
        loss_l1 = self.l1(pred_mag, target_mag)
        return self.alpha * loss_mrstft + (1 - self.alpha) * loss_l1


class CRMLoss(nn.Module):
    """Complex Ratio Mask loss on real/imag components."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_complex: torch.Tensor, target_complex: torch.Tensor) -> torch.Tensor:
        return self.mse(pred_complex.real, target_complex.real) + self.mse(pred_complex.imag, target_complex.imag)


def apply_crm_to_noisy(noisy_mag: torch.Tensor, noisy_phase: torch.Tensor, pred_mask: torch.Tensor, *,
                       use_log_mag: bool = True,
                       mask_activation: str = "tanh") -> torch.Tensor:
    """Apply complex ratio mask to noisy complex spectrum and return predicted complex spectrum.

    noisy_mag: (B, 1, F, T) or (B, F, T)
    noisy_phase: (B, F, T)
    pred_mask: (B, 2, F, T) -> [mask_real, mask_imag]
    """
    if pred_mask.dim() != 4 or pred_mask.size(1) != 2:
        raise ValueError(f"CRM expects model output shape (B,2,F,T), got {tuple(pred_mask.shape)}")

    if noisy_mag.dim() == 4:
        noisy_mag = noisy_mag.squeeze(1)

    if use_log_mag:
        noisy_mag = torch.expm1(noisy_mag)

    mask_real = pred_mask[:, 0]
    mask_imag = pred_mask[:, 1]
    if mask_activation == "tanh":
        mask_real = torch.tanh(mask_real)
        mask_imag = torch.tanh(mask_imag)

    noisy_complex = torch.polar(noisy_mag, noisy_phase)
    nr, ni = noisy_complex.real, noisy_complex.imag

    pred_real = mask_real * nr - mask_imag * ni
    pred_imag = mask_real * ni + mask_imag * nr

    return torch.complex(pred_real, pred_imag)


def build_target_complex(target_mag: torch.Tensor, target_phase: torch.Tensor, *, use_log_mag: bool = True) -> torch.Tensor:
    """Build target complex spectrum from magnitude + phase."""
    if target_mag.dim() == 4:
        target_mag = target_mag.squeeze(1)
    if use_log_mag:
        target_mag = torch.expm1(target_mag)
    return torch.polar(target_mag, target_phase)


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet denoiser")
    parser.add_argument('--loss', type=str, default=None, choices=['mse', 'l1', 'mrstft', 'hybrid', 'hybrid_l1', 'combined', 'crm'], help='Loss function to use (mse, l1, mrstft, hybrid, hybrid_l1, combined, crm)')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha value for Hybrid Loss (weight for MR-STFT). Default is from config.')
    parser.add_argument('--lambda1', type=float, default=None, help='Lambda1 for Combined Loss (weight for MSE). Default is from config.')
    parser.add_argument('--lambda2', type=float, default=None, help='Lambda2 for Combined Loss (weight for MR-STFT). Default is from config.')
    parser.add_argument('--lr', type=float, default=None, help='Override initial learning rate (useful for fine-tuning)')
    parser.add_argument('--use-scheduler', dest='use_scheduler', action='store_true', default=cfg.USE_SCHEDULER, help='Enable Learning Rate Scheduler')
    parser.add_argument('--no-scheduler', dest='use_scheduler', action='store_false', help='Disable Learning Rate Scheduler')
    parser.add_argument('--reset-lr', action='store_true', help='Reset learning rate to initial value (ignore checkpoint LR)')
    parser.add_argument('--reset-best-loss', dest='reset_best_loss', action='store_true', help='Reset best_val_loss to inf when resuming (use when switching loss function/stage so the new stage can save unet_best.pt). Implicitly enabled when --reset-lr is set.')
    parser.add_argument('--dynamic', action='store_true', help='Use on-the-fly dynamic mixing instead of precomputed tensors')
    parser.add_argument('--mask-mode', action='store_true', default=False, help='Use ratio-mask prediction (sigmoid) instead of direct magnitude. Eliminates residual hiss.')
    parser.add_argument('--epochs', '--num_epochs', type=int, default=cfg.NUM_EPOCHS, help='Number of epochs to train. When resuming, treated as additional epochs (e.g. resume from 20 + --epochs 7 = train until epoch 27). When starting fresh, this is the total.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (loads model+optimizer state)')
    parser.add_argument('--start-epoch', dest='start_epoch', type=int, default=None, help='Override the starting epoch (useful when resuming best.pt but you want to count from the last trained epoch, e.g. --start-epoch 31)')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Number of LR warmup epochs (0 to disable warmup). Skipped when resuming with --reset-lr.')
    
    # Use parse_known_args to handle Jupyter/Colab kernel arguments (like -f ...) gracefully
    args, _ = parser.parse_known_args()
    return args





def get_dataloaders(root: str, cfg_obj, dynamic: bool = False):
    if dynamic:
        from dataset import DynamicDenoisingDataset
        from sklearn.model_selection import train_test_split
        import os

        def collect_files_internal(directory, extensions=(".wav", ".mp3", ".webm")):
            files = []
            if not os.path.exists(directory):
                return []
            for r, _, filenames in os.walk(directory):
                for f in filenames:
                    if f.lower().endswith(extensions):
                        files.append(os.path.join(r, f))
            return sorted(files)

        # Search across all potential raw directories
        clean_dirs = [cfg_obj.CLEAN_EN_DIR, cfg_obj.CLEAN_NP_DIR, cfg_obj.CLEAN_EN_MP3_DIR, cfg_obj.CLEAN_NP_WEBM_DIR]
        clean_files = []
        print(f"Searching for clean audio files in: {clean_dirs}")
        for d in clean_dirs:
            if not os.path.exists(d):
                print(f"  Warning: Directory not found: {d}")
                continue
            found = collect_files_internal(d)
            clean_files.extend(found)
            print(f"  Found {len(found)} files in {d}")

        noise_files = collect_files_internal(cfg_obj.NOISE_DIR)

        if not clean_files:
            raise ValueError(f"No clean files (.wav, .mp3, .webm) found in: {clean_dirs}")
        if not noise_files:
            raise ValueError(f"No noise files found in {cfg_obj.NOISE_DIR}")

        train_clean, val_clean = train_test_split(clean_files, test_size=0.2, random_state=42)
        
        target_mode = "noisy" if cfg_obj.NOISE2NOISE else "clean"
        train_ds = DynamicDenoisingDataset(train_clean, noise_files, split="train", return_phase=True, target=target_mode, cfg=cfg_obj)
        val_ds = DynamicDenoisingDataset(val_clean, noise_files, split="val", return_phase=True, target=target_mode, cfg=cfg_obj)
        print(f"Dynamic Mixing: {len(train_clean)} train, {len(val_clean)} val samples")
    else:
        target_mode = "noisy" if cfg_obj.NOISE2NOISE else "clean"
        # We now request phase to allow waveform reconstruction for MR-STFT
        train_ds = DenoisingDataset(root, split="train", target=target_mode, return_phase=True, cfg=cfg_obj)
        val_ds = DenoisingDataset(root, split="val", target=target_mode, return_phase=True, cfg=cfg_obj)
        print(f"Static Mode: {len(train_ds)} train, {len(val_ds)} val samples")


    common_loader_kwargs = dict(
        batch_size=cfg_obj.BATCH_SIZE,
        num_workers=cfg_obj.NUM_WORKERS,
        pin_memory=cfg_obj.PIN_MEMORY,
        persistent_workers=cfg_obj.NUM_WORKERS > 0,
        prefetch_factor=2 if cfg_obj.NUM_WORKERS > 0 else None,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **{k: v for k, v in common_loader_kwargs.items() if v is not None},
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **{k: v for k, v in common_loader_kwargs.items() if v is not None},
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    use_mrstft = False
    use_hybrid = False
    use_combined = False
    if MultiResolutionSTFTLoss is not None:
        use_mrstft = isinstance(criterion, MultiResolutionSTFTLoss)
    use_hybrid = isinstance(criterion, (HybridLoss, HybridL1Loss))
    use_crm = isinstance(criterion, CRMLoss)
    if CombinedLoss is not None:
        use_combined = isinstance(criterion, CombinedLoss)

    print(f"  Starting data loading for {len(loader)} batches...")
    optimizer.zero_grad()
    for batch_idx, batch_data in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"  Training batch {batch_idx}/{len(loader)}...", end='\r', flush=True)
            
        # Unpack based on return_phase=True/False
        if len(batch_data) == 4:
            noisy_input, target_mag, noisy_phase, target_phase = batch_data
            noisy_phase = noisy_phase.to(device)
            target_phase = target_phase.to(device)
        else:
            noisy_input, target_mag = batch_data
            noisy_phase, target_phase = None, None

        noisy_input = noisy_input.to(device)
        target_mag = target_mag.to(device)

        pred = model(noisy_input)
        
        if use_crm:
            if noisy_phase is None or target_phase is None:
                raise ValueError("CRM training requires both noisy_phase and target_phase")
            pred_complex = apply_crm_to_noisy(
                noisy_input,
                noisy_phase,
                pred,
                use_log_mag=cfg.USE_LOG_MAG,
                mask_activation=cfg.CRM_MASK_ACTIVATION,
            )
            target_complex = build_target_complex(target_mag, target_phase, use_log_mag=cfg.USE_LOG_MAG)
            loss = criterion(pred_complex, target_complex)
        elif use_hybrid:
            pred_wav = reconstruct_waveform_from_mag_and_phase(pred, noisy_phase)
            target_wav = reconstruct_waveform_from_mag_and_phase(target_mag, target_phase)
            loss = criterion(pred, target_mag, pred_wav, target_wav)
        elif use_combined:
            pred_wav = reconstruct_waveform_from_mag_and_phase(pred, noisy_phase)
            target_wav = reconstruct_waveform_from_mag_and_phase(target_mag, target_phase)
            # Returns combined_loss, mse, mrstft. We only backward the combined loss
            loss, mse_val, mrstft_val = criterion(pred_wav, target_wav)
        elif use_mrstft:
            # Reconstruct waveforms for time-domain loss
            # Use noisy_phase for prediction (standard approximation)
            pred_wav = reconstruct_waveform_from_mag_and_phase(pred, noisy_phase)
            # Use target_phase for target (ground truth waveform)
            target_wav = reconstruct_waveform_from_mag_and_phase(target_mag, target_phase)
            loss = criterion(pred_wav, target_wav)
        else:
            loss = criterion(pred, target_mag)
            
        # Scale loss for accumulation
        loss = loss / cfg.ACCUMULATION_STEPS
        loss.backward()
        
        if (batch_idx + 1) % cfg.ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += (loss.item() * cfg.ACCUMULATION_STEPS) * noisy_input.size(0)
    print(f"  Training batch {len(loader)}/{len(loader)} - Complete!        ")
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    use_mrstft = False
    use_hybrid = False
    use_combined = False
    if MultiResolutionSTFTLoss is not None:
        use_mrstft = isinstance(criterion, MultiResolutionSTFTLoss)
    use_hybrid = isinstance(criterion, (HybridLoss, HybridL1Loss))
    use_crm = isinstance(criterion, CRMLoss)
    if CombinedLoss is not None:
        use_combined = isinstance(criterion, CombinedLoss)
        
    for batch_data in loader:
        if len(batch_data) == 4:
            noisy_input, target_mag, noisy_phase, target_phase = batch_data
            noisy_phase = noisy_phase.to(device)
            target_phase = target_phase.to(device)
        else:
            noisy_input, target_mag = batch_data
            noisy_phase, target_phase = None, None

        noisy_input = noisy_input.to(device)
        target_mag = target_mag.to(device)

        pred = model(noisy_input)
        
        if use_crm:
            if noisy_phase is None or target_phase is None:
                raise ValueError("CRM validation requires both noisy_phase and target_phase")
            pred_complex = apply_crm_to_noisy(
                noisy_input,
                noisy_phase,
                pred,
                use_log_mag=cfg.USE_LOG_MAG,
                mask_activation=cfg.CRM_MASK_ACTIVATION,
            )
            target_complex = build_target_complex(target_mag, target_phase, use_log_mag=cfg.USE_LOG_MAG)
            loss = criterion(pred_complex, target_complex)
        elif use_hybrid:
            pred_wav = reconstruct_waveform_from_mag_and_phase(pred, noisy_phase)
            target_wav = reconstruct_waveform_from_mag_and_phase(target_mag, target_phase)
            loss = criterion(pred, target_mag, pred_wav, target_wav)
        elif use_combined:
            pred_wav = reconstruct_waveform_from_mag_and_phase(pred, noisy_phase)
            target_wav = reconstruct_waveform_from_mag_and_phase(target_mag, target_phase)
            loss, *_ = criterion(pred_wav, target_wav)
        elif use_mrstft:
            pred_wav = reconstruct_waveform_from_mag_and_phase(pred, noisy_phase)
            target_wav = reconstruct_waveform_from_mag_and_phase(target_mag, target_phase)
            loss = criterion(pred_wav, target_wav)
        else:
            loss = criterion(pred, target_mag)
            
        running_loss += loss.item() * noisy_input.size(0)
    return running_loss / len(loader.dataset)


def main():
    args = parse_args()

    # Build a per-run immutable config with any CLI overrides
    global cfg
    overrides = {}
    if args.loss is not None:
        overrides["LOSS_FUNCTION"] = args.loss
    if args.alpha is not None:
        overrides["HYBRID_ALPHA"] = args.alpha
    if args.lambda1 is not None:
        overrides["COMBINED_LAMBDA1"] = args.lambda1
    if args.lambda2 is not None:
        overrides["COMBINED_LAMBDA2"] = args.lambda2
    if args.lr is not None:
        overrides["INITIAL_LR"] = args.lr
    if args.use_scheduler != cfg.USE_SCHEDULER:
        overrides["USE_SCHEDULER"] = args.use_scheduler
    if args.epochs != cfg.NUM_EPOCHS:
        overrides["NUM_EPOCHS"] = args.epochs
    if args.mask_mode:
        overrides["MASK_MODE"] = True

    cfg = get_config(overrides if overrides else None)

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            pass
        try:
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            pass

    # Set up logging to the logs/ folder
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    logger = setup_logger(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    data_root = cfg.PROCESSED_ROOT

    # Ensure checkpoint directory exists so we do not lose models after training
    checkpoint_dir = cfg.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Loading datasets...")
    train_loader, val_loader = get_dataloaders(data_root, cfg, dynamic=args.dynamic)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Logic for selecting the loss function
    # --loss argument overrides config
    loss_selection = cfg.LOSS_FUNCTION

    logger.info("Initializing model...")
    final_activation = "none" if loss_selection == "crm" else "softplus"
    use_mask_mode = cfg.MASK_MODE
    model = UNet(in_ch=cfg.UNET_IN_CH, out_ch=cfg.UNET_OUT_CH, base_ch=cfg.UNET_BASE_CH,
                 final_activation=final_activation, mask_mode=use_mask_mode).to(device)
    if use_mask_mode:
        logger.info("MASK MODE enabled — UNet predicts [0,1] ratio mask (sigmoid)")
    model.apply(init_kaiming)
    
    criterion = None
    use_mrstft_final = False

    if loss_selection == "hybrid":
        if MultiResolutionSTFTLoss is None:
            raise ImportError("MR_STFT module not found, cannot use hybrid loss")
        mrstft_loss = MultiResolutionSTFTLoss(
            fft_sizes=cfg.MR_STFT_FFT_SIZES,
            hop_sizes=cfg.MR_STFT_HOP_SIZES,
            win_lengths=cfg.MR_STFT_WIN_LENGTHS,
            window=cfg.MR_STFT_WINDOW
        ).to(device)
        
        alpha_val = cfg.HYBRID_ALPHA
        criterion = HybridLoss(mrstft_loss, alpha=alpha_val).to(device)
        logger.info(f"Using Hybrid Loss ({alpha_val} * MR-STFT + {1-alpha_val:.2f} * MSE)")

    elif loss_selection == "hybrid_l1":
        if MultiResolutionSTFTLoss is None:
            raise ImportError("MR_STFT module not found, cannot use hybrid_l1 loss")
        mrstft_loss = MultiResolutionSTFTLoss(
            fft_sizes=cfg.MR_STFT_FFT_SIZES,
            hop_sizes=cfg.MR_STFT_HOP_SIZES,
            win_lengths=cfg.MR_STFT_WIN_LENGTHS,
            window=cfg.MR_STFT_WINDOW
        ).to(device)
        alpha_val = cfg.HYBRID_ALPHA
        criterion = HybridL1Loss(mrstft_loss, alpha=alpha_val).to(device)
        logger.info(f"Using Hybrid L1 Loss ({alpha_val} * MR-STFT + {1-alpha_val:.2f} * L1) — robust to noisy targets")
            
    elif loss_selection == "combined":
        if CombinedLoss is None:
            raise ImportError("MR_STFT module not found, cannot use combined loss")
        
        l1 = cfg.COMBINED_LAMBDA1
        l2 = cfg.COMBINED_LAMBDA2
        logger.info(f"Using Combined Loss (L_total = {l1} * MSE + {l2} * MRSTFT)")
        
        criterion = CombinedLoss(
            lambda1=l1,
            lambda2=l2,
            fft_sizes=cfg.MR_STFT_FFT_SIZES,
            hop_sizes=cfg.MR_STFT_HOP_SIZES,
            win_lengths=cfg.MR_STFT_WIN_LENGTHS,
            window=cfg.MR_STFT_WINDOW
        ).to(device)

    elif loss_selection == "mrstft":
        if MultiResolutionSTFTLoss is None:
            raise ImportError("MR_STFT module not found, cannot use mrstft loss")
        criterion = MultiResolutionSTFTLoss(
            fft_sizes=cfg.MR_STFT_FFT_SIZES,
            hop_sizes=cfg.MR_STFT_HOP_SIZES,
            win_lengths=cfg.MR_STFT_WIN_LENGTHS,
            window=cfg.MR_STFT_WINDOW
        ).to(device)
        logger.info("Using Multi-Resolution STFT Loss (MR-STFT)")

    elif loss_selection == "crm":
        if cfg.UNET_OUT_CH != 2:
            raise ValueError("CRM requires UNET_OUT_CH=2 (real and imaginary mask channels)")
        criterion = CRMLoss().to(device)
        logger.info(f"Using CRM Loss (mask activation: {cfg.CRM_MASK_ACTIVATION})")

    elif loss_selection == "l1":
        criterion = nn.L1Loss().to(device)
        logger.info("Using L1 Loss (median-seeking, robust to noisy targets)")

    else:
        # Default to MSE
        criterion = nn.MSELoss().to(device)
        logger.info("Using MSE Loss")
        
    optimizer = None
    initial_lr = cfg.INITIAL_LR
    logger.info(f"Initial Learning Rate: {initial_lr}")
    wd = cfg.WEIGHT_DECAY if cfg.USE_WEIGHT_DECAY else 0.0
    
    if hasattr(torch.optim, "AdamW"):
        fused_ok = torch.cuda.is_available() and hasattr(torch.optim.AdamW, "fused")
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, betas=cfg.ADAM_BETAS, weight_decay=wd, fused=fused_ok)
            logger.info("Using fused AdamW optimizer" if fused_ok else "Using AdamW optimizer")
        except TypeError:
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=wd)
            logger.info("Fused AdamW unavailable; using Adam")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=wd)
        logger.info("AdamW not found; using Adam")

    # -- Resume checkpoint first (so we know start_epoch for scheduler) --
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info(f"Resuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            state_dict = ckpt.get("model_state_dict") if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            
            if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                
            if isinstance(ckpt, dict) and "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            if isinstance(ckpt, dict) and "best_val_loss" in ckpt:
                best_val_loss = ckpt["best_val_loss"]

            # Reset best_val_loss when starting a new stage (different loss function scale)
            # Triggered explicitly via --reset-best-loss or implicitly via --reset-lr
            if args.reset_best_loss or args.reset_lr:
                logger.info(f"Resetting best_val_loss from {best_val_loss:.4f} → inf (new stage, loss scale may differ)")
                best_val_loss = float("inf")

            # FORCED LR RESET Logic
            if args.reset_lr or args.lr is not None:
                new_lr = args.lr if args.lr is not None else initial_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                logger.warning(f"FORCING LEARNING RATE TO {new_lr}")

            logger.info(f"Resumed at epoch {start_epoch} with best_val_loss={best_val_loss:.4f}")
        else:
            logger.warning(f"Resume checkpoint not found at {args.resume}; starting fresh")

    # Allow manual override of start epoch (e.g. resume best.pt but continue from epoch 31)
    if args.start_epoch is not None:
        logger.info(f"Overriding start_epoch: {start_epoch} → {args.start_epoch} (via --start-epoch)")
        start_epoch = args.start_epoch

    # -- Build scheduler AFTER resume so it knows start_epoch --
    is_resuming = (args.resume is not None and start_epoch > 1)

    # Treat --epochs as *additional* epochs when resuming.
    # e.g. resume from epoch 20, --epochs 7 → train until epoch 27
    num_epochs = (start_epoch - 1 + args.epochs) if is_resuming else args.epochs

    scheduler = None
    if args.use_scheduler:
        import math as _math
        # When resuming with --reset-lr, skip warmup (already trained)
        warmup_epochs = 0 if (is_resuming and args.reset_lr) else args.warmup_epochs
        # Cosine annealing spans from start_epoch to num_epochs
        # so the LR decays smoothly over the REMAINING training, not the full run
        effective_start = start_epoch if is_resuming else 0
        effective_total = max(1, num_epochs - effective_start)
        min_lr_ratio = 1e-6 / initial_lr

        def lr_lambda(epoch):
            # epoch is 0-indexed within the scheduler (calls since creation)
            # Map to actual training progress
            actual_epoch = effective_start + epoch
            if epoch < warmup_epochs:
                # Linear warmup from 10% to 100%
                return 0.1 + 0.9 * float(epoch) / float(max(1, warmup_epochs))
            else:
                # Cosine annealing from 1 to min_lr_ratio
                progress_after_warmup = float(epoch - warmup_epochs) / float(max(1, effective_total - warmup_epochs))
                progress_after_warmup = min(progress_after_warmup, 1.0)
                return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + _math.cos(_math.pi * progress_after_warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if warmup_epochs > 0:
            logger.info(f"Scheduler: Linear Warmup ({warmup_epochs} epochs) + Cosine Annealing over {effective_total} epochs")
        else:
            logger.info(f"Scheduler: Cosine Annealing over {effective_total} remaining epochs (warmup skipped — resume with reset-lr)")
    else:
        logger.info("Learning Rate Scheduler DISABLED")

    if is_resuming:
        logger.info(f"Resuming from epoch {start_epoch - 1}. Running {args.epochs} additional epochs → training until epoch {num_epochs}.")
    else:
        logger.info(f"Starting training for {num_epochs} epochs...")
    epoch = start_epoch - 1  # initialize for interruption safety
    try:
        for epoch in range(start_epoch, num_epochs + 1):
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch}/{num_epochs} | LR: {current_lr:.2e}")
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)
            logger.info(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            # Update learning rate (LambdaLR steps on epoch, not on loss like ReduceLROnPlateau)
            if scheduler is not None:
                scheduler.step()

            # Save checkpoint for every epoch
            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            }
            ckpt_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch:02d}.pt")
            torch.save(ckpt_payload, ckpt_path)

            # Track the best validation loss and keep a convenient copy
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_payload["best_val_loss"] = best_val_loss
                best_ckpt_path = os.path.join(checkpoint_dir, "unet_best.pt")
                torch.save(ckpt_payload, best_ckpt_path)
                logger.info(f"New best checkpoint saved to {best_ckpt_path}")
    except KeyboardInterrupt:
        paused_path = os.path.join(checkpoint_dir, "unet_paused.pt")
        logger.warning("Training interrupted. Saving paused checkpoint...")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
        }, paused_path)
        logger.info(f"Paused checkpoint saved to {paused_path}")
        return

    logger.info("Saving final model state dict...")
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "unet_placeholder.pt"))
    logger.info(f"Training complete! Final model saved to {os.path.join(checkpoint_dir, 'unet_placeholder.pt')}")


if __name__ == "__main__":
    
    main()


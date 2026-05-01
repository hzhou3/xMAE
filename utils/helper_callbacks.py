import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import random
import math
import itertools
import torch
import os
import wandb
from typing import Dict, List, Tuple, Optional

class MaskedReconstruction(pl.Callback):
    def __init__(self, num_samples=4, every_n_epochs=5, outdir=None, mode='ppg'):
        """
        mode: 'ppg' or 'ecg' — which modality to mask & reconstruct in the plots
        """
        super().__init__()
        assert mode in ("ppg", "ecg"), "mode must be 'ppg' or 'ecg'"
        self.num_samples = int(num_samples)
        self.every_n_epochs = int(every_n_epochs)
        self.outdir = outdir
        self.mode = mode

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # only run every N epochs and only on global_zero
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return

        # pull a batch from val loader
        dl = trainer.datamodule.val_dataloader()
        # Generate a random integer between 0 and the number of batches
        num_batches = len(dl)
        random_batch_index = random.randint(0, num_batches - 1)
        dl_iterator = iter(dl)
        batch = next(iter(itertools.islice(dl_iterator, random_batch_index, None)))
        
        device = pl_module.device
        source = (getattr(pl_module, "source", "ppg+ecg") or "ppg+ecg").lower()

        # basic sizes
        fs = int(getattr(pl_module.hparams, "sampling_freq", 100))
        seg_len = int(getattr(pl_module.hparams, "seg_len", 10))
        P = pl_module.patch_len
        N = pl_module.n_patches
        L = seg_len * fs

        # get raw tensors on device (B,L) or (B,1,L) -> keep as (B,1,L) internally
        def _to_B1L(t):
            if t is None: return None
            t = t.to(device)
            return t.unsqueeze(1) if t.dim() == 2 else t

        ppg = _to_B1L(batch.get("ppg", None)) if source in {"ppg", "ppg+ecg"} else None
        ecg = _to_B1L(batch.get("ecg", None)) if source in {"ecg", "ppg+ecg"} else None
        pvc = batch.get("pvc", None) 
        pac = batch.get("pac", None) 


        # choose which modality to mask for visualization
        mask_ppg = (self.mode == "ppg") and (ppg is not None)
        mask_ecg = (self.mode == "ecg") and (ecg is not None)

        def _make_ids_for(mask_this: bool, B: int, ratio: float):
            if not mask_this or ratio <= 0.0:
                ids_all = torch.arange(N, device=device).view(1, N).expand(B, N)
                pm = torch.zeros(B, N, device=device)
                return ids_all.clone(), ids_all.clone(), N, pm
            pm = pl_module._make_patch_mask(B, N, ratio, pl_module.mask_mode)
            ids_keep, ids_restore, Nk = pl_module._ids_from_patch_mask(pm)
            return ids_keep, ids_restore, Nk, pm



        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            B = (ppg if ppg is not None else ecg).size(0)

            # derive ids per modality (mask only the selected one)
            ids_keep_ppg = ids_restore_ppg = Nk_ppg = None
            ids_keep_ecg = ids_restore_ecg = Nk_ecg = None
            
            
            if ppg is not None:
                ids_keep_ppg, ids_restore_ppg, Nk_ppg, pm_ppg = _make_ids_for(mask_ppg, B, pl_module.mask_ratio_ppg)
            if ecg is not None:
                ids_keep_ecg, ids_restore_ecg, Nk_ecg, pm_ecg = _make_ids_for(mask_ecg, B, pl_module.mask_ratio_ecg)
            

            
            # prepare model inputs as (B,L)
            def _to_BL(t):
                if t is None: return None
                return t.squeeze(1) if t.dim() == 3 else t

            outputs = pl_module.model(
                _to_BL(ppg), _to_BL(ecg),
                ids_keep_ppg=ids_keep_ppg, ids_restore_ppg=ids_restore_ppg,
                ids_keep_ecg=ids_keep_ecg, ids_restore_ecg=ids_restore_ecg,
            )

            # fetch reconstruction & seq mask for selected modality
            recon = outputs[f"{self.mode}_reconstructed"]         # (B,1,L) or (B,L)
            if recon.dim() == 2:
                recon = recon.unsqueeze(1)
                
            # Build seq mask from ids_restore + Nk (1 on masked)
            if self.mode == "ppg":
                seq_mask = pl_module._seqmask_from_ids_restore(ids_restore_ppg, Nk_ppg)  # (B,1,L)
                x = ppg
                x_another = ecg
            else:
                seq_mask = pl_module._seqmask_from_ids_restore(ids_restore_ecg, Nk_ecg)
                x = ecg
                x_another = ppg
                

            # compose full signal: recon on masked regions, original elsewhere
            reconstructed_full = recon * seq_mask + x * (1 - seq_mask)

        pl_module.train(was_training)

        # ensure output dir if saving
        if self.outdir is not None:
            os.makedirs(self.outdir, exist_ok=True)

        num_to_plot = min(self.num_samples, x.shape[0])
        tick_times = np.arange(0, seg_len + 1, 1)
        tick_indices = (tick_times * fs).astype(int)

        # detect logger
        exp = getattr(trainer.logger, "experiment", None)
        use_tb = hasattr(exp, "add_figure")
        use_wandb = (not use_tb) and hasattr(exp, "log")
        have_wandb = False
        if use_wandb:
            try:
                import wandb  # noqa: F401
                have_wandb = True
            except Exception:
                have_wandb = False

        for i in range(num_to_plot):
            try:
                pvc_count = pvc[i]
                pac_count = pac[i]
            except:
                pvc_count = 0
                pac_count = 0

            original_signal = x[i].squeeze().detach().cpu().numpy()
            recon_signal = reconstructed_full[i].squeeze().detach().cpu().numpy()
            mask_vis = seq_mask[i].squeeze().detach().cpu().numpy()

            x_second = x_another[i].squeeze().detach().cpu().numpy()

            # Create figure and two subplots sharing the x-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

            # --- Plotting on the first subplot (ax1) ---
            ax1.plot(original_signal, label='Original Signal', alpha=0.7)

            # Show reconstruction only on masked indices
            recon_only_masked = np.where(mask_vis == 1.0, recon_signal, np.nan)
            ax1.plot(recon_only_masked, label='Reconstructed Masked Region', linestyle=':', linewidth=2)

            # Shade masked spans
            masked_idx = np.where(mask_vis == 1.0)[0]
            if masked_idx.size > 0:
                start = masked_idx[0]
                end = start
                for idx in masked_idx[1:]:
                    if idx == end + 1:
                        end = idx
                    else:
                        ax1.axvspan(start, end + 1, color='red', alpha=0.25)
                        start, end = idx, idx
                ax1.axvspan(start, end + 1, color='red', alpha=0.25)

            ax1.set_title(f'Reconstruction of {self.mode.upper()}: Sample {i}', fontsize=18)
            ax1.set_ylabel('Amplitude', fontsize=16)
            ax1.legend(loc='upper right', fontsize=12)
            ax1.grid(True, alpha=0.3)

            # --- Plotting on the second subplot (ax2) ---
            ax2.plot(x_second, label='Another Signal')
            ax2.set_ylabel('Amplitude', fontsize=16)
            ax2.legend(loc='upper right', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Since the x-axis is shared, we only need to set ticks and labels on the bottom subplot (ax2)
            ax2.set_xticks(tick_indices, tick_times)
            ax2.set_xlabel('Time (seconds)', fontsize=16)

            plt.tight_layout()

            tag = f"Val_{self.mode.upper()}_Reconstructions/Sample_{i+1}_pvc{pvc_count}_pac{pac_count}"
            
            if self.outdir is not None:
                plt.savefig(os.path.join(self.outdir, f"{self.mode}_sample_{i+1}.png"))
            else:
                if use_tb:
                    exp.add_figure(tag, fig, global_step=trainer.global_step)
                elif use_wandb and have_wandb:
                    import wandb
                    exp.log({tag: wandb.Image(fig)}, step=trainer.global_step)
            plt.close(fig)



    
    
def get_callbacks(cfg):
    # map callback names to constructors
    callback_map = {
        
        "masked": lambda: MaskedReconstruction(
            num_samples=cfg["callback"]["n_plots"],
            every_n_epochs=cfg["callback"]["every_n_epochs"],
            mode=cfg["callback"].get('mode', 'ppg'),
        ),
    }

    # build requested callbacks
    callbacks = [callback_map[name]() for name in cfg["callback"]["names"] if name in callback_map]

    # always include checkpoint + early stopping
    callbacks += [
        
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=f"checkpoints/{cfg['experiment']}",
            save_top_k=1,
            save_last=True,
            filename="best-model-{epoch:03d}-{val_loss:.5f}",
        ),
        
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["train"]["patience"],
            mode="min",
        ),
    ]

        

        
    print(f"[INFO] number of callbacks: {len(callbacks)}")
    return callbacks






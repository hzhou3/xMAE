import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class UniversalTrainer(pl.LightningModule):

    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.model = model
        self.save_hyperparameters(config)

        # Source mode
        self.source = (self.hparams.get("source") or "ppg+ecg").lower()
        assert self.source in {"ppg", "ecg", "ppg+ecg"}
        
        self.model_type = self.hparams.get("model", 'na').lower()
        self.max_epochs = self.hparams.get("train", {})["epochs"]

        # Sizes
        fs = int(self.hparams.get("sampling_freq", 100))
        secs = int(self.hparams.get("seg_len", 10))
        self.seq_len = fs * secs

        # Patch length from model
        self.patch_len = getattr(getattr(self.model, "cfg", object()), "patch_len", 40)
        assert isinstance(self.patch_len, int) and self.patch_len > 0, "model.cfg.patch_len required"
        assert self.seq_len % self.patch_len == 0, "seq_len must be divisible by patch_len"
        self.n_patches = self.seq_len // self.patch_len

        # Loss weights
        lw = self.hparams.get("loss_weights", {}) or {}
        self.same_masks = bool(lw.get("same_masks", False))
        self.w_mwm_ppg = float(lw.get("mwm_ppg", 0.0))
        self.w_mwm_ecg = float(lw.get("mwm_ecg", 0.0))
        
        self.use_cl = bool(lw.get("use_cl", False))
        
        # Per-modality mask ratios
        self.mask_ratio_ppg = float(lw.get("mask_ratio_ppg", 0.0))
        self.mask_ratio_ecg = float(lw.get("mask_ratio_ecg", 0.8))  
        self.mask_mode = str(lw.get("mask_mode", "anchor"))  # "block" | "scatter" | "mixed" | "anchor"

        self.max_mask_ratio_ppg = self.mask_ratio_ppg
        self.max_mask_ratio_ecg = self.mask_ratio_ecg

        print("====")
        print(f"masking mode: {self.mask_mode}")
        print("====")


        self.cons_warmup_epochs = int(lw.get("cons_warmup_epochs", 0))  # epochs with weight 0
        self.cons_ramp_epochs = int(self.hparams.train.get("epoch", 50))  # epochs to ramp to max
        self.cons_schedule = str(lw.get("cons_schedule", "cosine"))  # 'linear' | 'cosine'

        # Optimizer params
        tr = self.hparams.get("train", {}) or {}
        self._opt_lr = float(tr.get("lr", 3e-5))
        self._opt_wd = float(tr.get("weight_decay", 0.01))

        # -----------------------------
        # loss-driven ladder curriculum state (minimal addition)
        # -----------------------------
        self._val_metric_sum = 0.0  # accumulate val metric per epoch
        self._val_metric_count = 0
        self._ladder_levels = None  # list of discrete mask ratios
        self._ladder_idx = 0
        self._ladder_ema = None
        self._ladder_start_ema = None
        self._ladder_epochs_at_level = 0
        self._ladder_pass_streak = 0
        self._ladder_min_epochs_per_level = 7  # will be set from epochs_per_level at first ladder call
        self._ladder_beta = 0.8         # EMA smoothing for val metric
        self._ladder_rel_improve = 0.10 # need 10% improvement vs level entry EMA to promote
        self._ladder_patience = 1       # consecutive passes required to promote
        self._ladder_abs_threshold = 0.09 # when to start ladder

    # -----------------------------
    # Masking utilities 
    # -----------------------------
    
    def _make_patch_mask(self, B: int, N: int, ratio: float, mode: str) -> torch.Tensor:
        """Return (B,N) with 1=masked, 0=keep.
        Supports full range: ratio ∈ [0, 1], including k == N (0 visible tokens).
        """
        device = self.device

        # Allow k up to N (was N-1)
        k = int(round(ratio * N))
        k = max(0, min(N, k))  # clamp to [0, N]
        V = N - k               # visible patches

        mode = mode if mode in {"block", "scatter", "mixed", "anchor"} else "anchor"

        # Fast paths
        if k == 0:
            return torch.zeros(B, N, device=device)  # nothing masked
        if k == N:
            return torch.ones(B, N, device=device)   # everything masked

        # ---- scatter: mask k random indices ----
        m_scatter = torch.zeros(B, N, device=device)
        idx = torch.rand(B, N, device=device).argsort(dim=1)[:, :k]
        m_scatter.scatter_(1, idx, 1.0)

        # ---- block: one contiguous MASKED block of length k ----
        m_block = torch.zeros(B, N, device=device)
        start_blk = torch.randint(0, N - k + 1, (B,), device=device)
        for b in range(B):
            m_block[b, start_blk[b] : start_blk[b] + k] = 1.0

        # ---- anchor: one contiguous VISIBLE block of length V at a RANDOM start ----
        # (i.e., exactly V zeros in one run; all others ones)
        m_anchor = torch.ones(B, N, device=device)
        if V > 0:
            start_vis = torch.randint(0, N - V + 1, (B,), device=device)
            for b in range(B):
                s = int(start_vis[b])
                m_anchor[b, s : s + V] = 0.0

        if mode == "block":
            return m_block
        if mode == "scatter":
            return m_scatter
        if mode == "anchor":
            return m_anchor

        # ---- mixed: 50/50 choose block or scatter per row ----
        choose_block = torch.rand(B, device=device) < 0.5
        out = torch.empty(B, N, device=device)
        for b in range(B):
            out[b] = m_block[b] if choose_block[b] else m_scatter[b]
        return out



    def change_mask_ratio_by_epoch(
        self,
        epoch: int,
        total_epochs: int,
        schedule: str = "ladder",
        min_ratio: float = 0.8,
        max_ratio: float = 0.9,
        step: float = 0.05, 
        epochs_per_level: int = 3,
        epochs_growth_per_level: float = 0.7,  # <- growth per level
        hold_first_mult: float = 1.0,  # stretch first level
        hold_last_mult: float = 2.0,  # stretch last level
        clamp: bool = True,
    ):
        # Levels (e.g., 0.60, 0.65, ..., 1.00)
        n_steps = int(round((max_ratio - min_ratio) / step))
        levels = [min_ratio + i * step for i in range(n_steps + 1)]
        L = len(levels)

        # NEW: loss-driven ladder → ignore epoch clock; just return current level
        if schedule == "ladder_loss":
            if (self._ladder_levels is None) or (self._ladder_levels != levels):
                # initialize ladder once from args (build from provided min/max/step)
                self._ladder_levels = levels
                self._ladder_idx = 0
                self._ladder_ema = None
                self._ladder_start_ema = None
                self._ladder_epochs_at_level = 0
                self._ladder_pass_streak = 0
                self._ladder_min_epochs_per_level = int(epochs_per_level)
            r_loss = float(self._ladder_levels[self._ladder_idx])
            return max(0.0, min(1.0, r_loss)) if clamp else r_loss
        else:
            print('Mask ratio does not change!!!')
            return min_ratio
        
        
        
    @staticmethod
    def _ids_from_patch_mask(patch_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        From patch_mask (B,N) with 1=masked, 0=keep:
          -> ids_keep (B,Nk), ids_restore (B,N), Nk (int)
        Keeps first in ascending order, then masks; ids_restore is inverse permutation.
        Enforces same Nk across batch.
        """
        B, N = patch_mask.shape
        device = patch_mask.device
        base_idx = torch.arange(N, device=device).view(1, N).expand(B, N)
        keep = patch_mask <= 0.0
        Nk_each = keep.sum(dim=1)
        assert int(Nk_each.min().item()) == int(Nk_each.max().item()), \
            "All items in the batch must have the same number of kept patches."
        Nk = int(Nk_each[0].item())
        scores = base_idx + (patch_mask > 0.0).float() * N  # keeps rank lower
        ids_sorted = torch.argsort(scores, dim=1)  # (B,N)
        ids_keep = ids_sorted[:, :Nk]  # (B,Nk)
        ids_restore = torch.argsort(ids_sorted, dim=1)  # (B,N)
        return ids_keep.long(), ids_restore.long(), Nk

    
    def _seqmask_from_ids_restore(self, ids_restore: torch.Tensor, Nk: int) -> torch.Tensor:
        """
        Build a (B,1,L) sequence mask with 1 on masked samples, 0 on kept.
        Uses the standard MAE trick: make a base mask [0]*Nk + [1]*(N-Nk) and "unshuffle" with ids_restore.
        """
        B, N = ids_restore.shape
        P = self.patch_len
        device = ids_restore.device

        # Base mask in the *sorted* space: kept first (0), then masked (1)
        base = torch.ones(B, N, device=device)
        if Nk > 0:
            base[:, :Nk] = 0.0

        # Unshuffle back to original patch order
        patch_mask = torch.gather(base, dim=1, index=ids_restore)  # (B,N)

        # Lift to (B,1,L) by repeating each patch mask over its P samples
        return patch_mask.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, P).view(B, 1, N * P)

    # -----------------------------
    # Model call (universal)
    # -----------------------------
    def _forward(self, batch):
        ppg = batch.get("ppg", None)
        ecg = batch.get("ecg", None)
        
        if self.source in {"ppg", "ppg+ecg"}:
            assert ppg is not None, "PPG data required for source including 'ppg'"
            if ppg.dim() == 3 and ppg.size(1) == 1:
                ppg = ppg.squeeze(1)
        if self.source in {"ecg", "ppg+ecg"}:
            assert ecg is not None, "ECG data required for source including 'ecg'"
            if ecg.dim() == 3 and ecg.size(1) == 1:
                ecg = ecg.squeeze(1)

        B = (ppg if ppg is not None else ecg).size(0)
        N = self.n_patches

        ids_keep_ppg = ids_restore_ppg = Nk_ppg = seq_mask_ppg = None
        ids_keep_ecg = ids_restore_ecg = Nk_ecg = seq_mask_ecg = None

        # Build masks/ids per active modality (per-modality ratios)
        pm_ppg = (
            self._make_patch_mask(B, N, self.mask_ratio_ppg, self.mask_mode)
            if self.source in {"ppg", "ppg+ecg"}
            else None
        )
        pm_ecg = (
            self._make_patch_mask(B, N, self.mask_ratio_ecg, self.mask_mode)
            if self.source in {"ecg", "ppg+ecg"}
            else None
        )

        if pm_ppg is not None:
            ids_keep_ppg, ids_restore_ppg, Nk_ppg = self._ids_from_patch_mask(pm_ppg)
            seq_mask_ppg = self._seqmask_from_ids_restore(ids_restore_ppg, Nk_ppg)
        if pm_ecg is not None:
            ids_keep_ecg, ids_restore_ecg, Nk_ecg = self._ids_from_patch_mask(pm_ecg)
            seq_mask_ecg = self._seqmask_from_ids_restore(ids_restore_ecg, Nk_ecg)

        outputs = self.model(
            ppg if self.source in {"ppg", "ppg+ecg"} else None,
            ecg if self.source in {"ecg", "ppg+ecg"} else None,
            ids_keep_ppg=ids_keep_ppg,
            ids_restore_ppg=ids_restore_ppg,
            ids_keep_ecg=ids_keep_ecg,
            ids_restore_ecg=ids_restore_ecg,
        )

        return (
            outputs,
            seq_mask_ppg,
            seq_mask_ecg,
            ids_keep_ppg,
            ids_restore_ppg,
            ids_keep_ecg,
            ids_restore_ecg,
        )

    # -----------------------------
    # Losses 
    # -----------------------------
    @staticmethod
    def _ensure_B1L(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.unsqueeze(1) if x.dim() == 2 else x

   
    def _calculate_loss(
        self,
        batch,
        outputs,
        seq_mask_ppg,
        seq_mask_ecg,
        *,
        ids_keep_ppg: Optional[torch.Tensor] = None,
        ids_restore_ppg: Optional[torch.Tensor] = None,
        run_consistency: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses. Optionally adds teacher→student consistency inside (training only).
        Pass the PPG ids so the student uses the SAME mask as the teacher.
        """
        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=self.device)

        ppg_orig = self._ensure_B1L(batch.get("ppg", None))
        ecg_orig = self._ensure_B1L(batch.get("ecg", None))
        ppg_rec = outputs.get("ppg_reconstructed", None)
        ecg_rec = outputs.get("ecg_reconstructed", None)
        # ppg_emb = outputs.get("ppg_embedding", None)
        # ecg_emb = outputs.get("ecg_embedding", None)
        
        
        # Masked waveform MSE
        if self.w_mwm_ppg > 0 and (ppg_rec is not None) and (ppg_orig is not None) and (seq_mask_ppg is not None):
            denom = seq_mask_ppg.sum().clamp_min(1.0)
            l = ((seq_mask_ppg * (ppg_rec - ppg_orig) ** 2).sum() / denom) * self.w_mwm_ppg
            losses["loss_mwm_ppg"] = l.detach()
            total += l

        if self.w_mwm_ecg > 0 and (ecg_rec is not None) and (ecg_orig is not None) and (seq_mask_ecg is not None):
            denom = seq_mask_ecg.sum().clamp_min(1.0)
            l = ((seq_mask_ecg * (ecg_rec - ecg_orig) ** 2).sum() / denom) * self.w_mwm_ecg
            losses["loss_mwm_ecg"] = l.detach()
            total += l


        losses["loss"] = total
        return losses

    # -----------------------------
    # Lightning hooks
    # -----------------------------
    def training_step(self, batch, batch_idx):
        # curriculum scheduling
        if self.use_cl:
            if self.max_mask_ratio_ppg > 0:
                # keep epoch-driven for PPG (or swap to "ladder_loss" similarly if desired)
                self.mask_ratio_ppg = self.change_mask_ratio_by_epoch(
                    self.current_epoch, self.max_epochs, min_ratio=self.mask_ratio_ppg
                )
            if self.max_mask_ratio_ecg > 0:
                # loss-driven ladder for ECG (ignores epoch clock)
                self.mask_ratio_ecg = self.change_mask_ratio_by_epoch(
                    self.current_epoch,
                    self.max_epochs,
                    schedule="ladder_loss",
                    min_ratio=self.mask_ratio_ecg,
                )

        (outputs, seq_mask_ppg, seq_mask_ecg, ids_keep_ppg, ids_restore_ppg, ids_keep_ecg, ids_restore_ecg) = self._forward(
            batch
        )

        losses = self._calculate_loss(
            batch,
            outputs,
            seq_mask_ppg,
            seq_mask_ecg,
            ids_keep_ppg=ids_keep_ppg,
            ids_restore_ppg=ids_restore_ppg,
            run_consistency=False,
        )
        
        # logging the mask ratios
        self.log("ppg_mask_ratio", self.mask_ratio_ppg, on_step=False, on_epoch=True)
        self.log("ecg_mask_ratio", self.mask_ratio_ecg, on_step=False, on_epoch=True)

        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, val in losses.items():
            self.log(
                f"{name}/train",
                val.detach() if torch.is_tensor(val) else val,
                on_epoch=True,
                sync_dist=True,
                prog_bar=(name == "loss"),
            )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        
        outputs, seq_mask_ppg, seq_mask_ecg, *_ = self._forward(batch)
        losses = self._calculate_loss(
            batch,
            outputs,
            seq_mask_ppg,
            seq_mask_ecg,
            ids_keep_ppg=None,
            ids_restore_ppg=None,
            run_consistency=False,  # disable student forward at val
        )

        self.log("val_loss", losses["loss"], on_epoch=True)
        for name, val in losses.items():
            self.log(
                f"{name}/val",
                val.detach() if torch.is_tensor(val) else val,
                on_epoch=True,
                sync_dist=True,
                prog_bar=(name == "loss"),
            )

        val_metric = losses.get("loss_mwm_ecg", losses["loss"])
        self._val_metric_sum += float(val_metric.detach().item())
        self._val_metric_count += 1

        return losses["loss"]
    

            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self._opt_lr, weight_decay=self._opt_wd, betas=(0.9, 0.95)
        )
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = steps_per_epoch * self.hparams["train"]["epochs"]
        warmup_steps = int(total_steps * self.hparams["train"].get("warmup_ratio", 0.1))

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def get_trainer(model, config: dict):
    return UniversalTrainer(model, config)



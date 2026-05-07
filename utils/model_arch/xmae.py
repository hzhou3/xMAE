from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# model config
# -----------------------------
@dataclass
class ModelCfg:
    seq_len: int = 1000
    patch_len: int = 40
    d_model: int = 512
    nhead: int = 8
    depth_ecg: int = 1
    depth_ppg: int = 2
    depth_bridge: int = 1
    stem_ch: int = 32
    dropout: float = 0.1
    proj_dim: int = 384
    use_cross_bridge: bool = True

    source: str = "ppg+ecg"          
    stem_type: str = "unet"          
    return_embeddings: bool = True   
    return_recon: bool = True      


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        if p is None: p = k // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU()
        )

class InceptionBlock1D(nn.Module):
    """Parallel small kernels + 1x1 fuse. Lightweight."""
    def __init__(self, in_ch, out_ch, ks=(3,5,7)):
        super().__init__()
        mid = out_ch // len(ks)
        mids = [mid]*(len(ks)-1) + [out_ch - mid*(len(ks)-1)]
        self.branches = nn.ModuleList([ConvBNAct(in_ch, m, k=k) for m,k in zip(mids, ks)])
        self.fuse = ConvBNAct(out_ch, out_ch, k=1, p=0)
    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], dim=1)
        return self.fuse(y)


class UNetHalfInceptionStem1D(nn.Module):
    """
    Down: L -> L/2 -> L/4 (Inception blocks)
    Up-fuse: L/4 -> L/2 -> L (FPN-like)
    Output length == input length. Residual skip from input.
    """
    def __init__(self, in_ch=1, out_ch=32, widths=(32,64,128)):
        super().__init__()
        c1, c2, c3 = widths
        self.enc1 = nn.Sequential(InceptionBlock1D(in_ch, c1), InceptionBlock1D(c1, c1))
        self.down1 = ConvBNAct(c1, c2, k=3, s=2)  # L/2
        self.enc2 = nn.Sequential(InceptionBlock1D(c2, c2), InceptionBlock1D(c2, c2))
        self.down2 = ConvBNAct(c2, c3, k=3, s=2)  # L/4
        self.enc3 = nn.Sequential(InceptionBlock1D(c3, c3), InceptionBlock1D(c3, c3))
        self.lat2 = nn.Conv1d(c2, c2, 1, bias=False)
        self.lat1 = nn.Conv1d(c1, c1, 1, bias=False)
        self.red3to2 = nn.Conv1d(c3, c2, 1, bias=False)
        self.red2to1 = nn.Conv1d(c2, c1, 1, bias=False)
        self.smooth2 = ConvBNAct(c2, c2, k=3)
        self.smooth1 = ConvBNAct(c1, c1, k=3)
        self.head = ConvBNAct(c1, out_ch, k=3)
        self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.skip_bn = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()
    def _upsample_to(self, x, L):
        return F.interpolate(x, size=L, mode="linear", align_corners=False)
    def forward(self, x):  # (B, in_ch, L)
        e1 = self.enc1(x)                     # (B,c1,L)
        e2 = self.enc2(self.down1(e1))        # (B,c2,L/2)
        e3 = self.enc3(self.down2(e2))        # (B,c3,L/4)
        t2 = self.lat2(e2) + self._upsample_to(self.red3to2(e3), e2.shape[-1])
        t2 = self.smooth2(t2)                 # (B,c2,L/2)
        t1 = self.lat1(e1) + self._upsample_to(self.red2to1(t2), e1.shape[-1])
        t1 = self.smooth1(t1)                 # (B,c1,L)
        y  = self.head(t1)                    # (B,out_ch,L)
        return y + self.skip_bn(self.skip(x))

class PatchEmbed1D(nn.Module):
    """Time → tokens using Conv1d with stride==kernel==patch_len."""
    def __init__(self, in_ch: int, d_model: int, patch_len: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Conv1d(in_ch, d_model, kernel_size=patch_len, stride=patch_len)
    def forward(self, x):  # (B,C,L)
        return self.proj(x).transpose(1, 2)  # (B, N, d)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerDecoderLite(nn.Module):
   
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):  # (B,N,D)
        h = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = h + x
        h = x
        x = self.ln2(x)
        x = h + self.mlp(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth: int, d_model: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerEncoderBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(depth)])
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class CrossAttentionBlock(nn.Module):
    """Update target tokens (q) by attending to source tokens (kv)."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.nq   = nn.LayerNorm(d_model)
        self.nkv  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
    def forward(self, q, kv):
        upd = self.attn(self.nq(q), self.nkv(kv), self.nkv(kv), need_weights=False)[0]
        x = q + upd
        x = x + self.ffn(x)
        return x

class WaveformDecoder(nn.Module):
    """Tokens → waveform via ConvTranspose1d."""
    def __init__(self, d_model: int, patch_len: int, out_ch: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(d_model, out_ch, kernel_size=patch_len, stride=patch_len)
    def forward(self, tokens):  # (B,N,d)
        x = tokens.transpose(1, 2)  # (B,d,N)
        return self.deconv(x)        # (B,1,L)


class SinglePath(nn.Module):
    """
        Process any B,1,L 
    """
    def __init__(
        self,
        *,
        seq_len: int,
        patch_len: int,
        d_model: int,
        proj_dim: int,
        nhead: int,
        depth: int,
        stem_ch: int,
        dropout: float,
        stem_type: str = "unet",      
        widths: Tuple[int, int, int] = (32, 64, 128),
    ):
        super().__init__()
        assert seq_len % patch_len == 0
        self.seq_len   = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.d_model   = d_model

        # stem
        if stem_type == "unet":
            self.stem = UNetHalfInceptionStem1D(in_ch=1, out_ch=stem_ch, widths=widths)
        else:
            raise ValueError(f"Unknown stem_type: {stem_type}")

        # patchify + pos
        self.patcher = PatchEmbed1D(stem_ch, d_model, patch_len)
        self.pos     = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # encoder / proj
        self.encoder = TransformerEncoder(depth=depth, d_model=d_model, nhead=nhead, dropout=dropout)
        self.proj    = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, proj_dim))

        # tiny decoder head for MAE
        self.dec_blk = TransformerDecoderLite(d_model, nhead, dropout=dropout)
        self.decoder = WaveformDecoder(d_model, patch_len, out_ch=1)
        
        self.mask_loc_table = nn.Embedding(self.n_patches, d_model)
        nn.init.xavier_uniform_(self.mask_loc_table.weight)

    @staticmethod
    def _ensure_B1L(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x

    def _identity_ids(self, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.arange(self.n_patches, device=device).long().view(1, -1).expand(B, -1)
        return idx, idx

    def visible_tokens_from_wave(self, x_B1L: torch.Tensor, ids_keep_BNk: torch.Tensor) -> torch.Tensor:
        """
        Gather visible waveform spans → (B,Nk,D).
        SAFE for Nk==0: returns empty (B,0,D) without running convs, but will not happen in our design.
        """
        B, _, L = x_B1L.shape
        P, N = self.patch_len, self.n_patches
        assert L == P * N
        Nk = ids_keep_BNk.size(1)

        if Nk == 0:
            # (B, 0, D) on correct device/dtype; avoid convs on empty signal
            return self.pos[:, :0].expand(B, 0, self.d_model).contiguous()

        # gather visible waveform
        x_bp = x_B1L.view(B, 1, N, P)
        idx  = ids_keep_BNk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, P)  # (B,1,Nk,P)
        x_vis = x_bp.gather(2, idx).reshape(B, 1, Nk * P)

        # encode
        t = self.patcher(self.stem(x_vis))  # (B,Nk,D)
        t = t + self.pos[:, :Nk]
        t = self.encoder(t)
        return t

    def decode_full_wave(self, tokens_vis: torch.Tensor,
                         ids_restore_BN: torch.Tensor,
                         mask_token_11D: torch.Tensor) -> torch.Tensor:
        """Restore to N with mask token + pos + tiny decode → waveform (B,1,L)."""
        B, Nk, D = tokens_vis.shape
        
        N = ids_restore_BN.size(1)
        
        if Nk != N:
            dec = torch.cat([tokens_vis, mask_token_11D.expand(B, N - Nk, D)], dim=1)   # (B,N,D)
            dec = torch.gather(dec, 1, ids_restore_BN.unsqueeze(-1).expand(-1, -1, D))  # reorder
            dec = dec + self.pos[:, :N]
        else:
            dec = tokens_vis
        
        dec = self.dec_blk(dec)
        return self.decoder(dec)

    @staticmethod
    def seqmask_from_ids_restore(ids_restore: torch.Tensor, Nk: int, patch_len: int) -> torch.Tensor:
        """
        Build a (B,1,L) sequence mask with 1 on masked samples, 0 on kept.
        """
        B, N = ids_restore.shape
        P = patch_len
        device = ids_restore.device

        base = torch.ones(B, N, device=device)
        if Nk > 0:
            base[:, :Nk] = 0.0

        patch_mask = torch.gather(base, dim=1, index=ids_restore)  # (B,N)
        return patch_mask.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, P).view(B, 1, N * P)


    def make_query_tokens_from_pre_encoded(
        self,
        t_vis_withpos: torch.Tensor,   # (B, Nk, D) from visible_tokens_from_wave
        ids_keep_BNk: torch.Tensor,    # (B, Nk)
        temp: torch.Tensor,    # (B, Nk)
    ) -> torch.Tensor:
        """
        Assemble full-length KV (B, N, D) for cross-attn:
          - Visible positions: use pre-encoded visibles (remove pos first).
          - Masked positions: use mask_loc_token.
        Finally add pos to all N (once).
        Works when Nk==0.
        """
        try:
            B, Nk, D = t_vis_withpos.shape
            
        except:
            B, _, D = temp.shape
            Nk = 0
            
        N = self.n_patches
            
        # remove pos so we can add it once uniformly
        t_vis_nopos = t_vis_withpos - self.pos[:, :Nk] if Nk > 0 else t_vis_withpos  # (B,Nk,D)

        # start with masked-location proxies
        q_full = self.mask_loc_table.weight.unsqueeze(0).expand(B, -1, -1).clone()

        # scatter visibles back to their original indices
        if Nk > 0:
            q_full.scatter_(1, ids_keep_BNk.unsqueeze(-1).expand(-1, -1, D), t_vis_nopos)

        # add pos to all positions exactly once
        q_full = q_full + self.pos[:, :N]
        return q_full  # (B,N,D)

    
    
    
    
class Bridger(nn.Module):
    """
    Cross-attention in both directions.
    - ECG→PPG: query = ECG, kv = PPG
    """
    def __init__(self, d_model: int, nhead: int, depth: int, dropout: float):
        super().__init__()
        self.ecg2ppg = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout=dropout)
            for _ in range(depth)
        ])
        self.ppg2ecg = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self,
                t_ppg: Optional[torch.Tensor],
                t_ecg: Optional[torch.Tensor],
                alpha: float = 1.0,
                *,
                q_ecg_override: Optional[torch.Tensor] = None,
                q_ppg_override: Optional[torch.Tensor] = None
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        # allow running when t_ecg is None/empty if we have an override for Q
        if (t_ppg is None) or (alpha <= 0.0):
            return t_ppg, (q_ecg_override if q_ecg_override is not None else t_ecg)

        t_ppg_hat = t_ppg 
        t_ecg_hat = q_ecg_override if (q_ecg_override is not None) else t_ecg
        if q_ppg_override is not None: # edge case but will not happen.
            t_ppg_hat = q_ppg_override

        # ECG→PPG: query = ECG, kv = PPG
        if (t_ecg_hat is not None) and (t_ppg is not None):
            for blk in self.ecg2ppg:
                upd = blk(t_ecg_hat, t_ppg)
                t_ecg_hat = t_ecg_hat + alpha * (upd - t_ecg_hat)

        return t_ppg_hat, t_ecg_hat



class xMAE(
    nn.Module,
):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.source = cfg.source.lower()
        assert self.source in {"ppg", "ecg", "ppg+ecg"}
        stem_type = cfg.stem_type

        L, P = cfg.seq_len, cfg.patch_len
        assert L % P == 0
        self.n_patches = L // P
        self.patch_len = P

        # modality paths
        self.ppg_path = None
        self.ecg_path = None

        if self.source in {"ppg", "ppg+ecg"}:
            self.ppg_path = SinglePath(
                seq_len=L, patch_len=P, d_model=cfg.d_model, proj_dim=cfg.proj_dim,
                nhead=cfg.nhead, depth=cfg.depth_ppg, stem_ch=cfg.stem_ch,
                dropout=cfg.dropout, stem_type=stem_type
            )
        if self.source in {"ecg", "ppg+ecg"}:
            self.ecg_path = SinglePath(
                seq_len=L, patch_len=P, d_model=cfg.d_model, proj_dim=cfg.proj_dim,
                nhead=cfg.nhead, depth=cfg.depth_ecg, stem_ch=cfg.stem_ch,
                dropout=cfg.dropout, stem_type=stem_type
            )

        # Bridger
        self.bridger = None
        if (self.source == "ppg+ecg") and cfg.use_cross_bridge:
            self.bridger = Bridger(cfg.d_model, cfg.nhead, cfg.depth_bridge, cfg.dropout)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)


    @staticmethod
    def _ensure_B1L(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x

    def _identity_ids(self, B: int, device):
        idx = torch.arange(self.n_patches, device=device).long().view(1, -1).expand(B, -1)
        return idx, idx

    def forward(self, 
                ppg=None,
                ecg=None,
                *,
                ids_keep_ppg=None, ids_restore_ppg=None,
                ids_keep_ecg=None, ids_restore_ecg=None,
                return_embeddings: Optional[bool] = None,
                return_recon: Optional[bool] = None):
        
        if return_embeddings is None:
            return_embeddings = self.cfg.return_embeddings
        if return_recon is None:
            return_recon = self.cfg.return_recon

        # Enforce source selection
        if self.source == "ppg":
            ecg = None
        elif self.source == "ecg":
            ppg = None

        assert (ppg is not None) or (ecg is not None), "At least one modality must be provided."

        if ppg is not None:
            ppg = self._ensure_B1L(ppg)
        if ecg is not None:
            ecg = self._ensure_B1L(ecg)

        ref = ppg if ppg is not None else ecg
        B, _, L = ref.shape
        assert L == self.cfg.seq_len
        device = ref.device

        # Identity ids if wasn't provided
        if (ppg is not None) and (ids_keep_ppg is None or ids_restore_ppg is None):
            ids_keep_ppg, ids_restore_ppg = self._identity_ids(B, device)
        if (ecg is not None) and (ids_keep_ecg is None or ids_restore_ecg is None):
            ids_keep_ecg, ids_restore_ecg = self._identity_ids(B, device)

        # Encode visible tokens per modality
        t_ppg = None
        t_ecg = None
        if ppg is not None:
            t_ppg = self.ppg_path.visible_tokens_from_wave(ppg, ids_keep_ppg)
        if ecg is not None:
            t_ecg = self.ecg_path.visible_tokens_from_wave(ecg, ids_keep_ecg)

        # visible ecg to full ecg with masks
        q_ecg_full = None
        if (self.bridger is not None) and (t_ppg is not None):
            q_ecg_full = self.ecg_path.make_query_tokens_from_pre_encoded(t_ecg, ids_keep_ecg, t_ppg)

        # Directional cross-attention 
        if ecg is not None and t_ecg is not None and self.bridger is not None:
            t_ppg_hat, t_ecg_hat = self.bridger(
                t_ppg, t_ecg,
                q_ecg_override=q_ecg_full
            )
        else:
            t_ppg_hat, t_ecg_hat = t_ppg, t_ecg
            
        
        out: Dict[str, torch.Tensor] = {}
        # Embeddings # for probing 
        if return_embeddings:
            if (t_ppg_hat is not None) and (t_ppg_hat.size(1) > 0):
                out["ppg_embedding"] = self.ppg_path.proj(t_ppg_hat.mean(dim=1))

        # Reconstructions
        if return_recon:
            if (t_ecg_hat is not None) and (ids_restore_ecg is not None):
                out["ecg_reconstructed"] = self.ecg_path.decode_full_wave(t_ecg_hat, ids_restore_ecg, self.mask_token)
                Nk = ids_keep_ecg.size(1)
                out["seq_mask_used_ecg"] = SinglePath.seqmask_from_ids_restore(ids_restore_ecg, Nk, self.patch_len)

        return out


def build_model_from_cfg(cfg_dict: dict) -> nn.Module:
    fs        = int(cfg_dict.get("sampling_freq", 100))
    seg_len_s = int(cfg_dict.get("seg_len", 10))
    seq_len   = fs * seg_len_s

    mp = cfg_dict.get("model_params", {}) 

    mcfg = ModelCfg(
        seq_len=seq_len,
        patch_len=int(mp.get("patch_len", 40)),
        d_model=int(mp.get("d_model", 512)),
        nhead=int(mp.get("nhead", 8)),
        depth_ecg=int(mp.get("depth_ecg", 6)),
        depth_ppg=int(mp.get("depth_ppg", 6)),
        depth_bridge=int(mp.get("depth_bridge", 2)),
        stem_ch=int(mp.get("stem_ch", 32)),
        dropout=float(mp.get("dropout", 0.1)),
        proj_dim=int(mp.get("latent_dim", 384)),
        use_cross_bridge=bool(mp.get("use_cross_bridge", True)),

        source=(cfg_dict.get("source") or "ppg+ecg").lower(),
        stem_type=str(mp.get("stem_type", "unet")),
        return_embeddings=True,
        return_recon=True,
    )

    return xMAE(mcfg)







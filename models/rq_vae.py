import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder_decoder import Encoder, Decoder


class RQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.codebook_size = cfg.codebook_size
        self.n_levels = cfg.rq_levels
        self.img_size = cfg.img_size
        self.channels = cfg.channels
        n_layers = len(cfg.encoder_channels)
        self.h = cfg.img_size // (2 ** n_layers)
        self.w = self.h
        self.encoder = Encoder(cfg.channels, cfg.encoder_channels, cfg.latent_dim)
        self.proj = nn.Conv2d(cfg.encoder_channels[-1], cfg.latent_dim, 1)
        self.decoder = Decoder(cfg.latent_dim, cfg.decoder_channels, cfg.channels)
        self.embeddings = nn.ModuleList([
            nn.Embedding(cfg.codebook_size, cfg.latent_dim) for _ in range(cfg.rq_levels)
        ])
        for emb in self.embeddings:
            emb.weight.data.uniform_(-1.0 / cfg.codebook_size, 1.0 / cfg.codebook_size)
        self.beta = getattr(cfg, "beta", 0.25)

    def encode(self, x):
        h = self.encoder(x)
        return self.proj(h)

    def quantize_level(self, r, emb):
        b, c, h, w = r.shape
        r_flat = r.permute(0, 2, 3, 1).reshape(-1, c)
        d = torch.cdist(r_flat, emb.weight)
        idx = d.argmin(dim=1)
        e = emb(idx).view(b, h, w, c).permute(0, 3, 1, 2)
        return e, idx

    def quantize(self, z):
        residuals = []
        codes = []
        r = z
        z_q = torch.zeros_like(z)
        for l in range(self.n_levels):
            e, idx = self.quantize_level(r, self.embeddings[l])
            residuals.append(r)
            codes.append(idx)
            z_q = z_q + e
            r = r - e
        return z_q, residuals, codes

    def decode(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.decoder(z)
        if out.size(2) != self.img_size:
            out = F.interpolate(out, (self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return out

    def forward(self, x):
        z = self.encode(x)
        z_q, residuals, codes = self.quantize(z)
        z_q_st = z + (z_q - z).detach()
        return self.decode(z_q_st), z, z_q, residuals, codes

    def loss(self, x):
        recon, z, z_q, residuals, codes = self.forward(x)
        rec_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        total = rec_loss
        codebook_sum = torch.tensor(0.0, device=recon.device)
        commit_sum = torch.tensor(0.0, device=recon.device)
        for l in range(self.n_levels):
            r = residuals[l]
            idx = codes[l]
            b, c, h, w = r.shape
            e = self.embeddings[l](idx).view(b, h, w, c).permute(0, 3, 1, 2)
            cb = nn.functional.mse_loss(e, r.detach(), reduction="mean")
            cm = nn.functional.mse_loss(r, e.detach(), reduction="mean")
            codebook_sum = codebook_sum + cb
            commit_sum = commit_sum + cm
            total = total + cb + self.beta * cm
        return total, rec_loss, codebook_sum, commit_sum
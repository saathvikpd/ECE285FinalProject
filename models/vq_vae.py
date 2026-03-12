import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder_decoder import Encoder, Decoder


class VQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.codebook_size = cfg.codebook_size
        self.img_size = cfg.img_size
        self.channels = cfg.channels
        n_layers = len(cfg.encoder_channels)
        self.h = cfg.img_size // (2 ** n_layers)
        self.w = self.h
        self.encoder = Encoder(cfg.channels, cfg.encoder_channels, cfg.latent_dim)
        self.proj = nn.Conv2d(cfg.encoder_channels[-1], cfg.latent_dim, 1)
        self.decoder = Decoder(cfg.latent_dim, cfg.decoder_channels, cfg.channels)
        self.embedding = nn.Embedding(cfg.codebook_size, cfg.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / cfg.codebook_size, 1.0 / cfg.codebook_size)
        self.beta = getattr(cfg, "beta", 0.25)

    def encode(self, x):
        h = self.encoder(x)
        return self.proj(h)

    def quantize(self, z):
        b, c, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        d = torch.cdist(z_flat, self.embedding.weight)
        idx = d.argmin(dim=1)
        z_q = self.embedding(idx).view(b, h, w, c).permute(0, 3, 1, 2)
        return z_q, idx

    def decode(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.decoder(z)
        if out.size(2) != self.img_size:
            out = F.interpolate(out, (self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return out

    def forward(self, x):
        z = self.encode(x)
        z_q, idx = self.quantize(z)
        z_q_st = z + (z_q - z).detach()
        return self.decode(z_q_st), z, z_q

    def loss(self, x):
        recon, z, z_q = self.forward(x)
        rec_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        commit = nn.functional.mse_loss(z, z_q.detach(), reduction="mean")
        codebook = nn.functional.mse_loss(z_q, z.detach(), reduction="mean")
        total = rec_loss + codebook + self.beta * commit
        return total, rec_loss, codebook, commit

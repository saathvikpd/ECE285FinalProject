import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder_decoder import Encoder, Decoder


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.img_size = cfg.img_size
        self.channels = cfg.channels
        self.encoder = Encoder(cfg.channels, cfg.encoder_channels, cfg.latent_dim)
        n_layers = len(cfg.encoder_channels)
        self.h = cfg.img_size // (2 ** n_layers)
        self.w = self.h
        self.feat_dim = cfg.encoder_channels[-1] * self.h * self.w
        self.fc_mu = nn.Linear(self.feat_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self.feat_dim, cfg.latent_dim)
        self.decoder = Decoder(cfg.latent_dim, cfg.decoder_channels, cfg.channels)
        self.beta = getattr(cfg, "beta", 1.0)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        z = z.repeat(1, 1, self.h, self.w)
        out = self.decoder(z)
        if out.size(2) != self.img_size:
            out = F.interpolate(out, (self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x):
        recon, mu, logvar = self.forward(x)
        rec_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        return rec_loss + (self.beta / self.latent_dim) * kl, rec_loss, kl

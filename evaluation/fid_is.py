import torch
import torch.nn.functional as F


def to_299(batch, channels, device):
    if batch.device != device:
        batch = batch.to(device)
    x = (batch + 1) / 2.0
    x = torch.clamp(x, 0, 1)
    if channels == 1:
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x


def generate_vae(model, n, device):
    z = torch.randn(n, model.latent_dim, device=device)
    return model.decode(z)


def generate_vq(model, n, device):
    if model.h == 1 and model.w == 1:
        idx = torch.randint(0, model.codebook_size, (n,), device=device)
        z_q = model.embedding(idx)
        return model.decode(z_q)
    b, h, w = n, model.h, model.w
    idx = torch.randint(0, model.codebook_size, (b * h * w,), device=device)
    z_q = model.embedding(idx).view(b, h, w, model.latent_dim).permute(0, 3, 1, 2)
    return model.decode(z_q)


def generate_rq(model, n, device):
    if model.h == 1 and model.w == 1:
        z_q = torch.zeros(n, model.latent_dim, device=device)
        for l in range(model.n_levels):
            idx = torch.randint(0, model.codebook_size, (n,), device=device)
            z_q = z_q + model.embeddings[l](idx)
        return model.decode(z_q)
    b, h, w = n, model.h, model.w
    z_q = torch.zeros(b, model.latent_dim, h, w, device=device)
    for l in range(model.n_levels):
        idx = torch.randint(0, model.codebook_size, (b * h * w,), device=device)
        e = model.embeddings[l](idx).view(b, h, w, model.latent_dim).permute(0, 3, 1, 2)
        z_q = z_q + e
    return model.decode(z_q)


def run_fid_is(model, model_name, val_loader, device, cfg):
    from ignite.engine import Engine
    from ignite.metrics import FID, InceptionScore

    channels = cfg.channels

    def step(engine, batch):
        x = batch[0].to(device)
        n = x.size(0)
        with torch.no_grad():
            if model_name == "vae":
                fake = generate_vae(model, n, device)
            elif model_name == "vq_vae":
                fake = generate_vq(model, n, device)
            else:
                fake = generate_rq(model, n, device)
        real_299 = to_299(x, channels, device)
        fake_299 = to_299(fake, channels, device)
        return real_299, fake_299

    evaluator = Engine(step)
    fid_metric = FID(device=device)
    is_metric = InceptionScore(device=device, output_transform=lambda x: x[1])
    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator, "is")

    class LimitedLoader:
        def __init__(self, loader, max_batches):
            self.loader = loader
            self.max_batches = max_batches
        def __iter__(self):
            for i, batch in enumerate(self.loader):
                if i >= self.max_batches:
                    break
                yield batch
        def __len__(self):
            return min(self.max_batches, len(self.loader))

    eval_loader = LimitedLoader(val_loader, cfg.fid_batches)
    evaluator.run(eval_loader, max_epochs=1)
    return evaluator.state.metrics["fid"], evaluator.state.metrics["is"]

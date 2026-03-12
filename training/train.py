import os
import torch
from torch.optim import Adam
from torchvision.utils import save_image

from data import build_mnist_loaders, build_abo_loaders
from models import VAE, VQVAE, RQVAE
from evaluation import metrics
from evaluation.fid_is import run_fid_is, generate_vae, generate_vq, generate_rq


def train(cfg, model_name, dataset_name):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if dataset_name == "mnist":
        train_loader, val_loader = build_mnist_loaders(cfg)
    else:
        train_loader, val_loader = build_abo_loaders(cfg)

    if model_name == "vae":
        model = VAE(cfg)
    elif model_name == "vq_vae":
        model = VQVAE(cfg)
    else:
        model = RQVAE(cfg)
    model = model.to(device)
    if getattr(torch, "compile", None) is not None:
        model = torch.compile(model, mode="reduce-overhead")
    opt = Adam(model.parameters(), lr=cfg.lr)

    os.makedirs(cfg.output_dir, exist_ok=True)
    img_dir = os.path.join(cfg.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, f"{dataset_name}_{model_name}.txt")
    log_file = open(log_path, "w")

    if cfg.use_wandb:
        import wandb
        run_name = cfg.wandb_run or f"{dataset_name}_{model_name}"
        wandb.init(project=f"{cfg.wandb_project}-{model_name}", name=run_name, config={
            "dataset": dataset_name, "model": model_name, "epochs": cfg.epochs,
            "batch_size": cfg.batch_size, "lr": cfg.lr, "seed": cfg.seed,
        })
        wandb.define_metric("epoch")
        wandb.define_metric("train_loss_breakdown/*", step_metric="epoch")
        wandb.define_metric("val_loss_breakdown/*", step_metric="epoch")

    track_loss_breakdown = model_name in ("vq_vae", "rq_vae")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_rec_sum = 0.0
        train_codebook_sum = 0.0
        train_commit_sum = 0.0
        train_n = 0
        for x, _ in train_loader:
            x = x.to(device)
            b = x.size(0)
            opt.zero_grad()
            loss, rec, *rest = model.loss(x)
            loss.backward()
            opt.step()
            train_loss += loss.item() * b
            if track_loss_breakdown:
                train_rec_sum += rec.item() * b
                train_codebook_sum += rest[0].item() * b
                train_commit_sum += rest[1].item() * b
            train_n += b
        train_loss /= train_n
        if track_loss_breakdown:
            train_rec_mean = train_rec_sum / train_n
            train_codebook_mean = train_codebook_sum / train_n
            train_commit_mean = train_commit_sum / train_n

        model.eval()
        val_loss = 0.0
        val_rec_sum = 0.0
        val_codebook_sum = 0.0
        val_commit_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                b = x.size(0)
                loss, rec, *rest = model.loss(x)
                val_loss += loss.item() * b
                if track_loss_breakdown:
                    val_rec_sum += rec.item() * b
                    val_codebook_sum += rest[0].item() * b
                    val_commit_sum += rest[1].item() * b
                val_n += b
        val_loss /= val_n
        if track_loss_breakdown:
            val_rec_mean = val_rec_sum / val_n
            val_codebook_mean = val_codebook_sum / val_n
            val_commit_mean = val_commit_sum / val_n

        log_dict = {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch}
        if track_loss_breakdown:
            log_dict["train_loss_breakdown/recon"] = train_rec_mean
            log_dict["train_loss_breakdown/codebook"] = train_codebook_mean
            log_dict["train_loss_breakdown/commit"] = train_commit_mean
            log_dict["val_loss_breakdown/recon"] = val_rec_mean
            log_dict["val_loss_breakdown/codebook"] = val_codebook_mean
            log_dict["val_loss_breakdown/commit"] = val_commit_mean

        if model_name == "vae":
            mus, logvars = [], []
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    mu, logvar = model.encode(x)
                    mus.append(mu)
                    logvars.append(logvar)
            mu_all = torch.cat(mus, dim=0)
            logvar_all = torch.cat(logvars, dim=0)
            kl = metrics.kl_divergence(mu_all, logvar_all)
            active = metrics.active_latent_dims(mu_all)
            proportion_active = active / cfg.latent_dim
            log_dict["val_kl"] = kl
            log_dict["Active dimensions"] = active
            log_dict["Proportion active dimensions"] = proportion_active
            line = f"epoch {epoch} train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_kl {kl:.4f} active_dims {active} proportion_active {proportion_active:.4f}\n"
        else:
            counts = metrics.gather_codebook_usage(
                model, val_loader, device, getattr(model, "n_levels", 1)
            )
            if len(counts) == 1:
                ent = metrics.codebook_entropy(counts[0])
                prop_used = metrics.codebook_proportion_used(counts[0])
                gini = metrics.gini_coefficient(counts[0])
                log_dict["Codebook entropy"] = ent
                log_dict["Codebook proportion used"] = prop_used
                log_dict["Codebook Gini"] = gini
                line = f"epoch {epoch} train_loss {train_loss:.4f} val_loss {val_loss:.4f} codebook_entropy {ent:.4f} codebook_proportion_used {prop_used:.4f} codebook_gini {gini:.4f}\n"
            else:
                ents = metrics.rq_level_entropy(counts)
                prop_used_list = [metrics.codebook_proportion_used(c) for c in counts]
                gini_list = [metrics.gini_coefficient(c) for c in counts]
                for i, e in enumerate(ents):
                    log_dict[f"Codebook entropy L{i}"] = e
                for i, p in enumerate(prop_used_list):
                    log_dict[f"Codebook proportion used L{i}"] = p
                for i, g in enumerate(gini_list):
                    log_dict[f"Codebook Gini L{i}"] = g
                line = f"epoch {epoch} train_loss {train_loss:.4f} val_loss {val_loss:.4f} rq_level_entropies {ents} codebook_proportion_used {prop_used_list} codebook_gini {gini_list}\n"
        log_file.write(line)
        log_file.flush()
        print(line.strip())

        if cfg.use_wandb:
            wandb.log(log_dict, step=epoch)

        if cfg.fid_every_n_epochs > 0 and epoch % cfg.fid_every_n_epochs == 0:
            model.eval()
            n_show = min(25, cfg.batch_size)
            with torch.no_grad():
                if model_name == "vae":
                    samples = generate_vae(model, n_show, device)
                elif model_name == "vq_vae":
                    samples = generate_vq(model, n_show, device)
                else:
                    samples = generate_rq(model, n_show, device)
            samples = (samples + 1) / 2.0
            samples = torch.clamp(samples, 0, 1)
            img_path = os.path.join(img_dir, f"{dataset_name}_{model_name}_{epoch}.png")
            save_image(samples, img_path, nrow=5, padding=2)

            fid, is_score = run_fid_is(model, model_name, val_loader, device, cfg)
            line = f"epoch {epoch} FID {fid:.4f} IS {is_score:.4f}\n"
            log_file.write(line)
            log_file.flush()
            print(line.strip())
            if cfg.use_wandb:
                wandb.log({"fid": fid, "inception_score": is_score, "epoch": epoch}, step=epoch)
                wandb.log({"generated": wandb.Image(img_path, caption=f"Epoch {epoch}")}, step=epoch)

    model.eval()
    if model_name == "vae":
        mus, logvars = [], []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                mu, logvar = model.encode(x)
                mus.append(mu)
                logvars.append(logvar)
        mu_all = torch.cat(mus, dim=0)
        logvar_all = torch.cat(logvars, dim=0)
        kl = metrics.kl_divergence(mu_all, logvar_all)
        active = metrics.active_latent_dims(mu_all)
        line = f"val_kl {kl:.4f} active_dims {active}\n"
    else:
        counts = metrics.gather_codebook_usage(
            model, val_loader, device, getattr(model, "n_levels", 1)
        )
        ent = metrics.codebook_entropy(counts[0]) if len(counts) == 1 else None
        if ent is not None:
            gini = metrics.gini_coefficient(counts[0])
            line = f"codebook_entropy {ent:.4f} codebook_gini {gini:.4f}\n"
        else:
            ents = metrics.rq_level_entropy(counts)
            gini_list = [metrics.gini_coefficient(c) for c in counts]
            line = f"rq_level_entropies {ents} codebook_gini {gini_list}\n"
    log_file.write(line)
    log_file.flush()
    print(line.strip())
    log_file.close()

    ckpt_path = os.path.join(cfg.output_dir, f"{dataset_name}_{model_name}.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    if cfg.use_wandb:
        import wandb
        wandb.finish()
    return model

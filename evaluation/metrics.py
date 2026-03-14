import torch


def gather_codebook_usage(model, loader, device):
    k = model.codebook_size
    counts = torch.zeros(k, device=device)
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            z = model.encode(x)
            _, idx = model.quantize(z)
            counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float, device=device))
    return [counts]


def kl_divergence(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return kl.mean().item()


def active_latent_dims(mu, eps=1e-2):
    var_per_dim = mu.var(dim=0)
    return (var_per_dim > eps).sum().item()


def codebook_entropy(usage_counts):
    n = usage_counts.sum().item()
    if n == 0:
        return 0.0
    p = usage_counts / n
    p = p[p > 0]
    return -(p * p.log()).sum().item()


def codebook_proportion_used(usage_counts):
    k = usage_counts.size(0)
    if k == 0:
        return 0.0
    return (usage_counts > 0).sum().item() / k


def gini_coefficient(usage_counts):
    k = usage_counts.size(0)
    if k == 0:
        return 0.0
    total = usage_counts.sum().item()
    if total == 0:
        return 0.0
    sorted_counts, _ = torch.sort(usage_counts)
    weighted_sum = sum((i + 1) * sorted_counts[i].item() for i in range(k))
    return (2.0 * weighted_sum - (k + 1) * total) / (k * total)

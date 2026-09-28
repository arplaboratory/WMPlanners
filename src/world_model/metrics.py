"""Image reconstruction metrics used by the optional DinoWM decoder."""

from functools import lru_cache

import torch
import torch.nn.functional as F


def l1_loss(network_output, gt):
    return torch.abs(network_output - gt).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    positions = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    weights = torch.exp(-(positions ** 2) / (2 * sigma ** 2))
    return weights / weights.sum()


def create_window(window_size, channel):
    one_dimensional = gaussian(window_size, 1.5).unsqueeze(1)
    two_dimensional = one_dimensional @ one_dimensional.T
    return two_dimensional.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel).to(
        device=img1.device, dtype=img1.dtype
    )
    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    def filtered(value):
        return F.conv2d(value, window, padding=window_size // 2, groups=channel)

    sigma1_sq = filtered(img1 * img1) - mu1_sq
    sigma2_sq = filtered(img2 * img2) - mu2_sq
    sigma12 = filtered(img1 * img2) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean((1, 2, 3))


def mse(img1, img2):
    return ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean()


def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2).clamp_min(1e-12)))


@lru_cache(maxsize=4)
def _lpips_model(device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS requires the optional 'decoder' package extra"
        ) from exc
    model = lpips.LPIPS(net="vgg").to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def eval_images(img1, img2, item_only=True, include_lpips=False):
    """Return image metrics; opt in to LPIPS after installing the decoder extra.

    Inputs are image tensors in [0, 1]. ``item_only`` is retained for callers
    of the previous API; all returned values are scalar tensors.
    """
    metrics = {
        "l1": l1_loss(img1, img2),
        "l2": l2_loss(img1, img2),
        "ssim": ssim(img1, img2),
        "mse": mse(img1, img2),
        "psnr": psnr(img1, img2),
    }
    if include_lpips:
        model = _lpips_model(img1.device)
        metrics["lpips"] = model(img1, img2, normalize=True).mean()
    return metrics

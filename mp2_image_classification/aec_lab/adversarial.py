"""Adversarial examples against the course model (FGSM / PGD, white-box).

No training and no labels: the attack asks the trained model for the gradient of its own loss
with respect to the input pixels, then moves every pixel a tiny step in the direction that hurts
the model most. `eps_255` bounds the change of every pixel (in 0–255 units); 2–8 is invisible.
"""
import io
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


def square(clf, image: Image.Image) -> Image.Image:
    """The photo exactly as the model sees it (resized + centre-cropped), still in normal colours."""
    return T.Compose([T.Resize(clf.size), T.CenterCrop(clf.size)])(image.convert("RGB"))


def _to_pil(x: torch.Tensor) -> Image.Image:
    a = (x[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


def _jpeg(img: Image.Image, quality: int = 75) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def attack(clf, image: Image.Image, eps_255: float = 4, steps: int = 10, target: Optional[str] = None,
           jpeg_quality: int = 75, seed: int = 0) -> Dict:
    """Returns the original (as seen by the model), the adversarial image, the amplified noise,
    the model's confidences before / after / after JPEG, and a few numbers for the summary.

    steps=1 is the one-shot Fast Gradient Sign Method; steps>1 is Projected Gradient Descent
    (with a random start), which is stronger for the same `eps_255`.
    """
    model, dev = clf.model, clf.device
    was_training = model.training
    model.eval()
    base = square(clf, image)
    x0 = T.ToTensor()(base).unsqueeze(0).to(dev)                 # [1,3,H,W] in [0,1]
    mean = torch.tensor(clf.mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(clf.std, device=dev).view(1, 3, 1, 1)

    def logits(x):
        return model(pixel_values=(x - mean) / std).logits.float()

    with torch.no_grad():
        y0 = int(logits(x0).argmax())
    eps = float(eps_255) / 255.0
    steps = max(1, int(steps))
    if target is not None and target in clf.classes:
        y_t = torch.tensor([clf.classes.index(target)], device=dev)
    else:
        y_t = None
    y_src = torch.tensor([y0], device=dev)

    g = torch.Generator(device="cpu").manual_seed(seed)
    if steps == 1:
        alpha, x = eps, x0.clone()
    else:
        alpha = max(eps / 4.0, 2.5 * eps / steps)
        x = (x0 + (torch.rand(x0.shape, generator=g) * 2 - 1).to(dev) * eps).clamp(0, 1)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        out = logits(x)
        # untargeted: make the model's own answer as unlikely as possible; targeted: make `target` likely
        loss = F.cross_entropy(out, y_src) if y_t is None else -F.cross_entropy(out, y_t)
        grad, = torch.autograd.grad(loss, x)
        x = x.detach() + alpha * grad.sign()
        x = torch.min(torch.max(x, x0 - eps), x0 + eps).clamp(0, 1)
    x_adv = (x.detach() * 255).round() / 255                    # a real 8-bit image, nothing hidden in fractions
    if was_training:
        model.train()

    adv = _to_pil(x_adv)
    noise = (x_adv - x0)                                          # in [-eps, eps]
    amp = 127.5 / max(float(eps_255), 1e-6)
    noise_img = _to_pil((noise * (0.5 / max(eps, 1e-6)) + 0.5).clamp(0, 1))   # grey = no change, full range = ±eps
    jpeg = _jpeg(adv, jpeg_quality)

    # all verdicts come from the classifier exactly as the rest of the notebook uses it
    p_before, p_after, p_jpeg = clf.predict_one(base), clf.predict_one(adv), clf.predict_one(jpeg)
    linf_255 = float((x_adv - x0).abs().max()) * 255
    return {
        "original": base, "adversarial": adv, "noise": noise_img, "jpeg": jpeg,
        "before": p_before, "after": p_after, "after_jpeg": p_jpeg,
        "source_class": clf.classes[y0], "target": target, "eps_255": float(eps_255), "steps": steps,
        "linf_255": linf_255, "amplification": amp, "jpeg_quality": jpeg_quality,
    }

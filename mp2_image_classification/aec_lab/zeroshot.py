"""Zero-shot classification with CLIP: students type the class names themselves."""
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .models import get_device

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class ZeroShot:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        import transformers
        from transformers import CLIPModel, CLIPProcessor
        self.device = get_device()
        old = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            self.model = CLIPModel.from_pretrained(model_id).eval().to(self.device)
            self.proc = CLIPProcessor.from_pretrained(model_id)
        finally:
            transformers.logging.set_verbosity(old)
        self.name = "CLIP (zero-shot)"

    @torch.no_grad()
    def predict(self, images: Sequence[Image.Image], class_names: Sequence[str], template: str = "a photo of {}") -> np.ndarray:
        texts = [template.format(c) for c in class_names]
        out = []
        for b in range(0, len(images), 16):
            inputs = self.proc(text=texts, images=[im.convert("RGB") for im in images[b:b + 16]],
                               return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**inputs).logits_per_image
            out.append(logits.softmax(-1).float().cpu().numpy())
        return np.concatenate(out)


def parse_classes(text: str) -> List[str]:
    parts = [p.strip() for chunk in text.replace(";", "\n").replace(",", "\n").split("\n") for p in [chunk]]
    return [p for p in parts if p]


def zero_shot_grid(zs: ZeroShot, images: Sequence[Image.Image], class_names: Sequence[str],
                   captions: Optional[Sequence[str]] = None, ncols: int = 4):
    from . import ui
    probs = zs.predict(images, class_names)
    titles = []
    for i in range(len(images)):
        j1, j2 = np.argsort(-probs[i])[:2]
        line = f"best: {class_names[j1]} ({probs[i, j1] * 100:.0f}%)\nthen: {class_names[j2]} ({probs[i, j2] * 100:.0f}%)"
        titles.append((captions[i] + "\n" if captions else "") + line)
    ui.show(ui.image_grid(list(images), titles, ncols=ncols, size=2.8,
                          suptitle="CLIP's best guess (and runner-up) using YOUR class names"))
    return probs

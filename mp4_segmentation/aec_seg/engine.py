"""SAM 3 (Segment Anything with Concepts) wrapper: phrase prompts and box prompts, nothing else exposed."""
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

MIRROR = os.environ.get("AEC_SEG_MODEL_ID", "jetjodh/sam3")   # public mirror of facebook/sam3 (same weights, SAM License,
                                                             # no gating); the env var lets tests point at a local copy
MAX_SIDE = 1280


@dataclass
class SegResult:
    """Instances found for one prompt on one image (masks at the image's own resolution)."""
    prompt: str
    masks: np.ndarray                 # [n, H, W] bool
    scores: np.ndarray                # [n]
    boxes: np.ndarray                 # [n, 4] xyxy pixels
    size: tuple                       # (W, H)
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.scores)

    def union(self, min_score: float = 0.0) -> np.ndarray:
        keep = self.scores >= min_score
        if not keep.any():
            return np.zeros((self.size[1], self.size[0]), dtype=bool)
        return self.masks[keep].any(0)

    def area_pct(self, min_score: float = 0.0) -> float:
        return float(self.union(min_score).mean() * 100.0)

    def label_map(self, min_score: float = 0.0) -> np.ndarray:
        """uint8 map: 0 = background, k = instance k (1-based, ordered by score)."""
        lab = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        order = np.argsort(-self.scores)
        k = 0
        for i in order:
            if self.scores[i] < min_score:
                continue
            k += 1
            lab[(self.masks[i]) & (lab == 0)] = min(k, 255)
        return lab

    @staticmethod
    def from_label_map(prompt: str, lab: np.ndarray, scores: Sequence[float], boxes: Sequence[Sequence[float]]) -> "SegResult":
        n = len(scores)
        masks = np.stack([lab == (k + 1) for k in range(n)]) if n else np.zeros((0,) + lab.shape, dtype=bool)
        return SegResult(prompt, masks, np.array(scores, dtype=float), np.array(boxes, dtype=float).reshape(-1, 4),
                         (lab.shape[1], lab.shape[0]))


def fit(image: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    im = image.convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


class Sam3Engine:
    def __init__(self, model_id: str = MIRROR, device: Optional[str] = None, log=print):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        t0 = time.time()
        log(f"Loading SAM 3 ({'GPU' if self.device == 'cuda' else 'CPU: each phrase will take 10-30 s'})...")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        if self.device == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        log(f"SAM 3 ready in {time.time() - t0:.0f} s.")
        self._torch = torch

    def _run(self, image: Image.Image, threshold: float, mask_threshold: float = 0.5, **prompt_kwargs) -> dict:
        torch = self._torch
        inputs = self.processor(images=image, return_tensors="pt", **prompt_kwargs).to(self.device)
        inputs = {k: (v.to(self.dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_instance_segmentation(out, threshold=threshold, mask_threshold=mask_threshold,
                                                                target_sizes=[image.size[::-1]])[0]
        return res

    def segment(self, image: Image.Image, text: str, threshold: float = 0.5,
                negative_boxes: Optional[Sequence[Sequence[float]]] = None) -> SegResult:
        """All objects matching the phrase. `negative_boxes` (xyxy, pixels) tell the model what NOT to include."""
        t0 = time.time()
        kw = {"text": text}
        if negative_boxes:
            kw["input_boxes"] = [[list(map(float, b)) for b in negative_boxes]]
            kw["input_boxes_labels"] = [[0] * len(negative_boxes)]
        res = self._run(image, threshold, **kw)
        return self._pack(text, res, image.size, time.time() - t0)

    def segment_box(self, image: Image.Image, box_xyxy: Sequence[float], threshold: float = 0.2) -> SegResult:
        """One object indicated by a box; keeps the highest-scoring instance that overlaps the box."""
        t0 = time.time()
        res = self._run(image, threshold, input_boxes=[[list(map(float, box_xyxy))]], input_boxes_labels=[[1]])
        packed = self._pack(f"box {[int(v) for v in box_xyxy]}", res, image.size, time.time() - t0)
        if len(packed) > 1:
            x1, y1, x2, y2 = box_xyxy
            bx = packed.boxes
            inter = np.maximum(0, np.minimum(bx[:, 2], x2) - np.maximum(bx[:, 0], x1)) * np.maximum(0, np.minimum(bx[:, 3], y2) - np.maximum(bx[:, 1], y1))
            area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]) + 1e-6
            ok = inter / area > 0.5
            cand = np.where(ok)[0] if ok.any() else np.arange(len(packed))
            best = cand[np.argmax(packed.scores[cand])]
            packed = SegResult(packed.prompt, packed.masks[best:best + 1], packed.scores[best:best + 1], packed.boxes[best:best + 1],
                               packed.size, packed.seconds)
        return packed

    def segment_like(self, image: Image.Image, box_xyxy: Sequence[float], threshold: float = 0.3) -> SegResult:
        """Every object that looks like the one inside the box: the box is a visual example, not a phrase."""
        t0 = time.time()
        res = self._run(image, threshold, input_boxes=[[list(map(float, box_xyxy))]], input_boxes_labels=[[1]])
        return self._pack(f"like box {[int(v) for v in box_xyxy]}", res, image.size, time.time() - t0)

    @staticmethod
    def _pack(prompt, res, size, seconds) -> SegResult:
        n = len(res["masks"])
        if n == 0:
            return SegResult(prompt, np.zeros((0, size[1], size[0]), dtype=bool), np.zeros(0), np.zeros((0, 4)), size, seconds)
        masks = res["masks"].bool().cpu().numpy() if hasattr(res["masks"], "cpu") else np.asarray(res["masks"]).astype(bool)
        scores = res["scores"].float().cpu().numpy() if hasattr(res["scores"], "cpu") else np.asarray(res["scores"], dtype=float)
        boxes = res["boxes"].float().cpu().numpy() if hasattr(res["boxes"], "cpu") else np.asarray(res["boxes"], dtype=float)
        order = np.argsort(-scores)
        return SegResult(prompt, masks[order], scores[order], boxes[order], size, seconds)

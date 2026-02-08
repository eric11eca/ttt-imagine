from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class CrossClipCoherenceMetrics:
    """
    Cross-clip coherence metrics for long-video generation.

    The implementation is intentionally lightweight:
    - If OpenAI CLIP isn't installed, CLIP-based metrics return NaN.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self._clip_model = None

    def _load_clip(self) -> None:
        if self._clip_model is not None:
            return
        try:
            import clip  # type: ignore

            self._clip_model, _ = clip.load("ViT-B/32", device=self.device)
            self._clip_model.eval()
        except Exception:
            self._clip_model = None

    @torch.no_grad()
    def clip_consistency(self, clips: list[torch.Tensor]) -> float:
        """
        Average cosine similarity between the last frame of clip i and first frame of clip i+1.
        """
        self._load_clip()
        if self._clip_model is None or len(clips) < 2:
            return float("nan")

        from torchvision.transforms import CenterCrop, Compose, Normalize, Resize

        preprocess = Compose(
            [
                Resize(224, antialias=True),
                CenterCrop(224),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        similarities = []
        for i in range(len(clips) - 1):
            last_frame = preprocess(clips[i][-1]).unsqueeze(0).to(self.device)
            first_frame = preprocess(clips[i + 1][0]).unsqueeze(0).to(self.device)

            feat_last = self._clip_model.encode_image(last_frame)
            feat_first = self._clip_model.encode_image(first_frame)

            feat_last = feat_last / feat_last.norm(dim=-1, keepdim=True)
            feat_first = feat_first / feat_first.norm(dim=-1, keepdim=True)

            sim = (feat_last @ feat_first.T).item()
            similarities.append(sim)

        return sum(similarities) / len(similarities)

    @torch.no_grad()
    def feature_drift(self, clips: list[torch.Tensor]) -> float:
        """
        1 - cosine similarity between a mid-frame feature of first and last clip.
        Lower is better.
        """
        self._load_clip()
        if self._clip_model is None or len(clips) < 2:
            return float("nan")

        from torchvision.transforms import CenterCrop, Compose, Normalize, Resize

        preprocess = Compose(
            [
                Resize(224, antialias=True),
                CenterCrop(224),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        feats = []
        for clip in clips:
            mid = clip[clip.shape[0] // 2]
            frame = preprocess(mid).unsqueeze(0).to(self.device)
            feat = self._clip_model.encode_image(frame)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat)

        sim = (feats[0] @ feats[-1].T).item()
        return 1.0 - sim

    def boundary_smoothness(self, clips: list[torch.Tensor]) -> float:
        """
        Average MSE between last frame of clip i and first frame of clip i+1 (pixel space).
        Lower is smoother.
        """
        if len(clips) < 2:
            return float("nan")

        distances = []
        for i in range(len(clips) - 1):
            last = clips[i][-1].float()
            first = clips[i + 1][0].float()
            distances.append(F.mse_loss(last, first).item())
        return sum(distances) / len(distances)


from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class NovelClipSequenceDataset(Dataset):
    """
    Dataset for E2E-TTT meta-learning.

    Each sample is a SEQUENCE of (text_segment, clip_latent) pairs representing one
    novel split into chronological clips.

    Expected precomputed directory layout:

        precomputed_clips_dir/
        └── <novel_id_md5>/
            ├── clip_000.pt
            ├── clip_001.pt
            ├── ...
            └── segments.json   # [{"text": "...", "clip_file": "clip_000.pt"}, ...]
    """

    def __init__(
        self,
        metadata_csv: str,
        precomputed_clips_dir: str,
        max_clips_per_novel: int = 20,
    ):
        with open(metadata_csv, "r", encoding="utf-8") as f:
            self.samples = list(csv.DictReader(f))
        self.precomputed_dir = precomputed_clips_dir
        self.max_clips = int(max_clips_per_novel)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples[idx]
        video_path = row.get("video_path", "")
        novel_id = hashlib.md5(video_path.encode()).hexdigest()
        clip_dir = os.path.join(self.precomputed_dir, novel_id)

        segments_path = os.path.join(clip_dir, "segments.json")
        with open(segments_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        segments = segments[: self.max_clips]

        clip_latents = []
        clip_texts = []
        for seg in segments:
            latent = torch.load(
                os.path.join(clip_dir, seg["clip_file"]),
                map_location="cpu",
            )
            clip_latents.append(latent.unsqueeze(0))  # [1, C, T', H', W']
            clip_texts.append(seg["text"])

        return {
            "clip_latents": clip_latents,
            "clip_texts": clip_texts,
            "novel_id": novel_id,
            "num_clips": len(clip_latents),
        }


def precompute_clip_latents(
    metadata_csv: str,
    vae: torch.nn.Module,
    output_dir: str,
    frames_per_clip: int = 17,
    height: int = 480,
    width: int = 832,
    novel_text_splitter: Optional[Callable[[str, int], list[str]]] = None,
    device: str = "cuda",
):
    """
    Offline preprocessing:
      - Split each video into clips
      - Encode each clip via a frozen VAE to z_0
      - Split novel text into `num_clips` segments
      - Save clips + `segments.json` in the directory structure expected by
        `NovelClipSequenceDataset`.

    This function is optional and not used by unit tests.
    """
    from torchvision.io import read_video
    import torchvision.transforms as T

    os.makedirs(output_dir, exist_ok=True)
    vae = vae.to(device).eval()

    with open(metadata_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    transform = T.Compose(
        [
            T.Resize((height, width), antialias=True),
            T.ConvertImageDtype(torch.float32),
        ]
    )

    for row in rows:
        video_path = row["video_path"]
        novel_text = row.get("novel_text", "")
        novel_id = hashlib.md5(video_path.encode()).hexdigest()
        clip_dir = os.path.join(output_dir, novel_id)
        segments_json = os.path.join(clip_dir, "segments.json")

        if os.path.exists(segments_json):
            continue

        os.makedirs(clip_dir, exist_ok=True)

        # Load video (T, H, W, C) uint8 -> (T, C, H, W) float [0,1]
        video_frames, _, _ = read_video(video_path, pts_unit="sec")
        video_frames = video_frames.permute(0, 3, 1, 2).float() / 255.0
        video_frames = torch.stack([transform(f) for f in video_frames])
        total_frames = video_frames.shape[0]

        num_clips = max(1, total_frames // frames_per_clip)

        # Split text into segments
        if novel_text_splitter is not None:
            text_segments = novel_text_splitter(novel_text, num_clips)
        else:
            sentences = novel_text.replace(".", ".\n").split("\n")
            sentences = [s.strip() for s in sentences if s.strip()]
            per_clip = max(1, len(sentences) // num_clips) if len(sentences) > 0 else 1
            text_segments = []
            for i in range(num_clips):
                start = i * per_clip
                end = start + per_clip if i < num_clips - 1 else len(sentences)
                text_segments.append(" ".join(sentences[start:end]) if sentences else "")

        segments = []
        for i in range(num_clips):
            frame_start = i * frames_per_clip
            frame_end = min(frame_start + frames_per_clip, total_frames)
            clip_frames = video_frames[frame_start:frame_end]

            if clip_frames.shape[0] < frames_per_clip:
                pad = torch.zeros(frames_per_clip - clip_frames.shape[0], 3, height, width)
                clip_frames = torch.cat([clip_frames, pad], dim=0)

            # VAE expects [1, T, C, H, W] or similar depending on implementation
            clip_batch = clip_frames.unsqueeze(0).to(device)
            with torch.no_grad():
                z_0 = vae.encode(clip_batch * 2.0 - 1.0).sample()

            clip_file = f"clip_{i:03d}.pt"
            torch.save(z_0.cpu().squeeze(0), os.path.join(clip_dir, clip_file))

            text_seg = text_segments[i] if i < len(text_segments) else text_segments[-1]
            segments.append({"clip_file": clip_file, "text": text_seg})

        with open(segments_json, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)

        logger.info("Preprocessed %s: %d clips", novel_id, num_clips)


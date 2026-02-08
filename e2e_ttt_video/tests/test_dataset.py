import csv
import hashlib
import json

import torch
import pytest

from e2e_ttt_video.dataset import NovelClipSequenceDataset


@pytest.fixture
def mock_clip_dataset(tmp_path):
    """Create a minimal clip-sequence dataset on disk."""
    csv_path = tmp_path / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_path", "novel_text", "caption_summary"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_path": "/fake/video1.mp4",
                "novel_text": "The hero walked. The hero ran. The hero stopped.",
                "caption_summary": "A hero's journey",
            }
        )

    novel_id = hashlib.md5(b"/fake/video1.mp4").hexdigest()
    clip_dir = tmp_path / "clips" / novel_id
    clip_dir.mkdir(parents=True)

    for i in range(3):
        latent = torch.randn(4, 2, 4, 4)  # [C, T', H', W']
        torch.save(latent, clip_dir / f"clip_{i:03d}.pt")

    segments = [
        {"clip_file": f"clip_{i:03d}.pt", "text": f"Segment {i}"} for i in range(3)
    ]
    with open(clip_dir / "segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f)

    return str(csv_path), str(tmp_path / "clips")


def test_dataset_length(mock_clip_dataset):
    csv_path, clips_dir = mock_clip_dataset
    ds = NovelClipSequenceDataset(csv_path, clips_dir)
    assert len(ds) == 1


def test_dataset_getitem_structure(mock_clip_dataset):
    csv_path, clips_dir = mock_clip_dataset
    ds = NovelClipSequenceDataset(csv_path, clips_dir)
    sample = ds[0]

    assert "clip_latents" in sample
    assert "clip_texts" in sample
    assert "num_clips" in sample
    assert sample["num_clips"] == 3
    assert len(sample["clip_latents"]) == 3
    assert len(sample["clip_texts"]) == 3


def test_dataset_clip_shapes(mock_clip_dataset):
    csv_path, clips_dir = mock_clip_dataset
    ds = NovelClipSequenceDataset(csv_path, clips_dir)
    sample = ds[0]

    for lat in sample["clip_latents"]:
        assert lat.dim() == 5  # [1, C, T', H', W']
        assert lat.shape[0] == 1


def test_dataset_max_clips(mock_clip_dataset):
    csv_path, clips_dir = mock_clip_dataset
    ds = NovelClipSequenceDataset(csv_path, clips_dir, max_clips_per_novel=2)
    sample = ds[0]
    assert sample["num_clips"] == 2


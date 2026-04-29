"""Inspect video metadata: resolution, FPS, frame count, duration, codec.

Single file:
    python -m data_utils.inspect_video path/to/video.mp4

Whole directory (one row per video):
    python -m data_utils.inspect_video path/to/videos/

Recursive scan + aggregated summary (uniq resolutions, fps, frame counts):
    python -m data_utils.inspect_video path/to/dataset/ --recursive --summary
"""

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from os import path as osp
from typing import List, Optional

import imageio


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    duration_sec: float
    num_frames: int
    codec: Optional[str] = None
    pix_fmt: Optional[str] = None
    file_size_mb: float = 0.0
    error: Optional[str] = None  # populated if probing failed


def probe_video(video_path: str, count_frames: bool = False) -> VideoMetadata:
    """Read header metadata via imageio's ffmpeg backend.

    `count_frames=False` (default) computes frame count as round(duration * fps),
    which is fast and accurate for constant-frame-rate videos. Set True to force
    a full decode pass (slow but exact, useful for variable-frame-rate sources).
    """
    try:
        reader = imageio.get_reader(video_path, "ffmpeg")
        meta = reader.get_meta_data()
        width, height = meta["size"]  # imageio reports (W, H)
        fps = float(meta["fps"])
        duration = float(meta["duration"])
        if count_frames:
            num_frames = reader.count_frames()
        else:
            num_frames = int(round(duration * fps))
        codec = meta.get("codec")
        pix_fmt = meta.get("pix_fmt")
        reader.close()
        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            duration_sec=duration,
            num_frames=num_frames,
            codec=codec,
            pix_fmt=pix_fmt,
            file_size_mb=osp.getsize(video_path) / (1024 ** 2),
        )
    except Exception as e:
        return VideoMetadata(
            path=video_path,
            width=0, height=0, fps=0.0, duration_sec=0.0, num_frames=0,
            error=f"{type(e).__name__}: {e}",
        )


def collect_videos(root: str, recursive: bool, ext: str = ".mp4") -> List[str]:
    """List all video files at `root`, optionally recursing into subdirs."""
    if osp.isfile(root):
        return [root]
    if not osp.isdir(root):
        raise FileNotFoundError(root)
    videos: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(ext):
                    videos.append(osp.join(dirpath, f))
    else:
        videos = [
            osp.join(root, f) for f in os.listdir(root)
            if f.endswith(ext) and osp.isfile(osp.join(root, f))
        ]
    return sorted(videos)


def format_row(m: VideoMetadata) -> str:
    if m.error:
        return f"  ERROR  {m.path}\n         {m.error}"
    return (
        f"  {m.width:>4} x {m.height:<4}  "
        f"{m.fps:>6.2f} fps  "
        f"{m.num_frames:>5}f  "
        f"{m.duration_sec:>6.2f}s  "
        f"{m.codec or '?':<6}  "
        f"{m.file_size_mb:>6.2f} MB  "
        f"{m.path}"
    )


def print_summary(metas: List[VideoMetadata]) -> None:
    """Aggregate distributions across many probes to surface inconsistencies."""
    ok = [m for m in metas if m.error is None]
    bad = [m for m in metas if m.error is not None]

    print(f"\n=== Summary: {len(metas)} videos ({len(ok)} OK, {len(bad)} failed) ===")
    if not ok:
        for m in bad:
            print(format_row(m))
        return

    res_counts = Counter((m.width, m.height) for m in ok)
    fps_counts = Counter(round(m.fps, 3) for m in ok)
    nframes_counts = Counter(m.num_frames for m in ok)
    codec_counts = Counter(m.codec for m in ok)

    def _print_dist(name: str, counter: Counter) -> None:
        print(f"\n  {name}:")
        for value, count in counter.most_common():
            print(f"    {count:>5}  {value}")

    _print_dist("Resolution (W x H)", res_counts)
    _print_dist("FPS", fps_counts)
    _print_dist("Frame count", nframes_counts)
    _print_dist("Codec", codec_counts)

    durations = [m.duration_sec for m in ok]
    sizes_mb = [m.file_size_mb for m in ok]
    print(f"\n  Duration:  min={min(durations):.2f}s  "
          f"max={max(durations):.2f}s  "
          f"mean={sum(durations)/len(durations):.2f}s")
    print(f"  Size:      min={min(sizes_mb):.2f}MB  "
          f"max={max(sizes_mb):.2f}MB  "
          f"total={sum(sizes_mb):.2f}MB")

    if bad:
        print(f"\n  Failed files ({len(bad)}):")
        for m in bad:
            print(f"    {m.path}: {m.error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=str, help="Video file or directory")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories")
    parser.add_argument("--summary", action="store_true",
                        help="Print aggregated distributions instead of per-file rows")
    parser.add_argument("--count-frames", action="store_true",
                        help="Decode each file fully to count frames exactly "
                             "(slow; default uses duration * fps)")
    parser.add_argument("--ext", type=str, default=".mp4",
                        help="File extension to match when scanning a directory")
    args = parser.parse_args()

    videos = collect_videos(args.path, recursive=args.recursive, ext=args.ext)
    if not videos:
        print(f"No '{args.ext}' files found under {args.path}")
        return

    metas = [probe_video(v, count_frames=args.count_frames) for v in videos]

    if args.summary:
        print_summary(metas)
    else:
        header = (
            f"  {'WxH':>11}  {'fps':>10}  {'frames':>6}  "
            f"{'dur':>6}  {'codec':<6}  {'size':>9}  path"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for m in metas:
            print(format_row(m))


if __name__ == "__main__":
    main()

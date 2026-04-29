"""Re-encode videos in a folder to a uniform target FPS.

Bridges heterogeneous-FPS datasets (caught by `inspect_video --summary`) into the
fixed-FPS contract that `precomp_video.py` requires (it asserts video_fps == fps).

Usage:
    python -m data_utils.resample_fps \\
        --input_dir path/to/videos \\
        --output_dir path/to/resampled \\
        --target_fps 16

    # Recursive scan, mirror subdirectory layout:
    python -m data_utils.resample_fps \\
        --input_dir path/to/dataset \\
        --output_dir path/to/resampled \\
        --target_fps 16 --recursive

    # Also resize:
    python -m data_utils.resample_fps ... --width 1280 --height 720

    # Skip ffmpeg (just copy) for files whose FPS already matches:
    python -m data_utils.resample_fps ... --copy-matching
"""

import argparse
import os
import shutil
import subprocess
from os import path as osp
from typing import List, Optional

from imageio_ffmpeg import get_ffmpeg_exe
from tqdm import tqdm

from data_utils.inspect_video import probe_video


def build_ffmpeg_cmd(
    input_path: str,
    output_path: str,
    target_fps: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    crf: int = 18,
    preset: str = "medium",
) -> List[str]:
    """Compose an ffmpeg command line for FPS resampling.

    `-r <fps>` placed AFTER `-i` is the *output* frame rate. ffmpeg drops or
    duplicates frames as needed (no temporal interpolation). `-crf 18` is
    visually-lossless x264; lower = higher quality (and larger files).
    """
    cmd = [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", input_path]
    if width is not None and height is not None:
        cmd += ["-vf", f"scale={width}:{height}"]
    cmd += [
        "-r", str(target_fps),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-an",  # drop audio: video-diffusion datasets are visual-only
        output_path,
    ]
    return cmd


def resample_one(
    input_path: str,
    output_path: str,
    target_fps: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    crf: int = 18,
    skip_if_exists: bool = True,
    copy_if_already_match: bool = False,
) -> str:
    """Resample one video. Returns a status string ('resampled', 'copied', ...).

    Raises on ffmpeg failure (caller decides whether to log-and-continue or abort).
    """
    if skip_if_exists and osp.exists(output_path):
        return "skipped (exists)"

    os.makedirs(osp.dirname(output_path), exist_ok=True)

    # Fast path: if input fps already matches and no resize requested, just copy.
    if copy_if_already_match and width is None and height is None:
        meta = probe_video(input_path)
        if meta.error is None and round(meta.fps, 3) == round(float(target_fps), 3):
            shutil.copy2(input_path, output_path)
            return "copied (fps already matches)"

    cmd = build_ffmpeg_cmd(input_path, output_path, target_fps, width, height, crf)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Remove partial output so a re-run does not see stale corrupt data.
        if osp.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.strip()}")

    return "resampled"


def collect_videos(root: str, recursive: bool, ext: str = ".mp4") -> List[str]:
    """List all `ext` files at `root`. Accepts either a file or a directory."""
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


def relative_output(video_path: str, input_root: str) -> str:
    """Path of `video_path` relative to `input_root`, or basename if root is a file."""
    if osp.isfile(input_root):
        return osp.basename(video_path)
    return osp.relpath(video_path, input_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_dir", required=True,
                        help="Source directory (or single video file)")
    parser.add_argument("--output_dir", required=True,
                        help="Destination directory; mirrors --input_dir layout")
    parser.add_argument("--target_fps", type=int, required=True,
                        help="Output frame rate. Lower = drops frames; higher = duplicates frames.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories of --input_dir")
    parser.add_argument("--width", type=int, default=None,
                        help="Optional resize width (must be set with --height)")
    parser.add_argument("--height", type=int, default=None,
                        help="Optional resize height (must be set with --width)")
    parser.add_argument("--crf", type=int, default=18,
                        help="x264 CRF (lower = higher quality; 18 ≈ visually lossless)")
    parser.add_argument("--preset", default="medium",
                        help="x264 preset: ultrafast..veryslow (slower = better compression)")
    parser.add_argument("--ext", default=".mp4")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode even if the output file already exists")
    parser.add_argument("--copy-matching", action="store_true",
                        help="If a file's fps already matches --target_fps and no resize "
                             "is requested, copy it instead of re-encoding (lossless, fast)")
    args = parser.parse_args()

    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be specified together")

    videos = collect_videos(args.input_dir, recursive=args.recursive, ext=args.ext)
    if not videos:
        print(f"No '{args.ext}' files found under {args.input_dir}")
        return

    print(f"Resampling {len(videos)} video(s) -> {args.target_fps} fps")
    if args.width is not None:
        print(f"  Resizing to {args.width}x{args.height}")

    counts = {
        "resampled": 0,
        "skipped (exists)": 0,
        "copied (fps already matches)": 0,
        "failed": 0,
    }
    for video in tqdm(videos):
        out_path = osp.join(args.output_dir, relative_output(video, args.input_dir))
        try:
            status = resample_one(
                input_path=video,
                output_path=out_path,
                target_fps=args.target_fps,
                width=args.width,
                height=args.height,
                crf=args.crf,
                skip_if_exists=not args.overwrite,
                copy_if_already_match=args.copy_matching,
            )
            counts[status] = counts.get(status, 0) + 1
        except Exception as e:
            counts["failed"] += 1
            print(f"FAILED {video}: {e}")

    print("\nResult:")
    for status, count in counts.items():
        print(f"  {count:>5}  {status}")


if __name__ == "__main__":
    main()

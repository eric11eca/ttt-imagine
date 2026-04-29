import argparse
import os
from os import path as osp

import imageio
import torch
import torch.distributed as dist
import torchvision.transforms as TT
from tqdm import tqdm

from videotuna.models.wan.wan.modules.vae import WanVAE, WanVAE_

def pad_video(frames: torch.Tensor, target_num_frames: int) -> torch.Tensor:
    """Symmetrically pad a (T, C, H, W) video tensor along the time dimension.

    Splits the deficit evenly between the front and back, repeating the first
    frame on the left and the last frame on the right. The left side gets the
    smaller half when the deficit is odd.
    """
    pad_num = target_num_frames - frames.shape[0]
    pad_left = pad_num // 2
    pad_right = pad_num - pad_left
    left = frames[:1].expand(pad_left, *frames.shape[1:])
    right = frames[-1:].expand(pad_right, *frames.shape[1:])
    return torch.cat([left, frames, right], dim=0)

def crop_video(frames: torch.Tensor, target_num_frames: int) -> torch.Tensor:
    """Center-crop a (T, C, H, W) video tensor along the time dimension."""
    overflow = frames.shape[0] - target_num_frames
    start = overflow // 2
    return frames[start : start + target_num_frames]

def get_vae(
    vae_weight_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> WanVAE:
    """Build and load the WAN 2.1 VAE wrapper.

    Architecture params match `configs/008_wanvideo/*.yaml::first_stage_config`.
    The wrapper's `encode` returns the scaled posterior mean (z_dim=16 channels)
    as float32; we cast to `dtype` at save time for storage efficiency.
    """
    vae_inner = WanVAE_(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    vae = WanVAE(vae=vae_inner, vae_pth=vae_weight_path, dtype=dtype, device=device)
    vae.load_weight()
    vae.model.eval()
    return vae

def precompute_episode(
    videos_dir: str,
    save_dir: str,
    vae: WanVAE,
    target_num_frames: int,
    fps: int,
    height: int,
    width: int,
) -> None:
    """Precompute WAN VAE latents for every .mp4 in `videos_dir`.

    Saves one .pt per video in `save_dir` with shape (z_dim, T_lat, H_lat, W_lat),
    where T_lat = target_num_frames // 4 + 1 (causal-conv temporal compression of 4x
    plus a boundary frame), and H_lat/W_lat = height/8, width/8 (8x spatial compression).
    """
    assert osp.isdir(videos_dir), f"Input directory does not exist: {videos_dir}"
    os.makedirs(save_dir, exist_ok=True)

    device = vae.device
    dtype = vae.dtype
    z_dim = vae.model.z_dim
    target_latent_frames = target_num_frames // 4 + 1
    h_latent = height // 8
    w_latent = width // 8
    expected_latent_shape = (z_dim, target_latent_frames, h_latent, w_latent)

    videos = sorted(v for v in os.listdir(videos_dir) if v.endswith(".mp4"))

    for video in tqdm(videos):
        video_path = osp.join(videos_dir, video)
        save_path = osp.join(save_dir, video.replace(".mp4", ".pt"))

        # Skip if already processed and valid
        if osp.exists(save_path):
            try:
                t = torch.load(save_path, map_location=torch.device('cpu'))
                assert t.shape == expected_latent_shape
                assert t.dtype == dtype
                assert t.min() > -10 and t.max() < 10, "latent out of range"
                continue
            except Exception as e:
                print(f"Redoing {video} due to error: {e}")

        # Read frames
        video_reader = imageio.get_reader(video_path, "ffmpeg")
        video_fps = video_reader.get_meta_data()["fps"]
        assert video_fps == fps, f"Video FPS ({video_fps}) does not match expected FPS ({fps})"

        frames = [TT.ToTensor()(frame) for frame in video_reader]
        video_reader.close()
        video = torch.stack(frames)  # (T, C, H, W) in [0, 1]

        if video.shape[0] < target_num_frames:
            video = pad_video(video, target_num_frames)
        elif video.shape[0] > target_num_frames:
            video = crop_video(video, target_num_frames)
        assert video.shape[0] == target_num_frames

        # (T, C, H, W) -> (C, T, H, W), [0, 1] -> [-1, 1]
        video = video.permute(1, 0, 2, 3).to(device).to(dtype).contiguous()
        video = video * 2.0 - 1.0
        assert video.shape == (3, target_num_frames, height, width)

        # WanVAE.encode takes a list of (C, T, H, W) tensors and returns a list of
        # scaled-mean latents in float32. We process one at a time because the
        # wrapper iterates internally — there is no real batching benefit here.
        with torch.no_grad():
            latent = vae.encode([video])[0]
        latent = latent.to(dtype).cpu().contiguous()
        assert latent.shape == expected_latent_shape
        assert latent.min() > -10 and latent.max() < 10, "latent out of range"
        torch.save(latent, save_path)

def init_distributed_if_available() -> tuple[int, int, int]:
    """Initialize torch.distributed if launched under torchrun, else single-process.

    Returns (rank, world_size, local_rank). Outside a distributed launch,
    returns (0, 1, 0) and never touches the process group.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0
    return rank, world_size, local_rank


def main():
    """Precompute WAN VAE latents for a directory tree of episode/clip mp4s."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, required=True,
                        help="Directory of subdirectories, each containing .mp4 clips")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Mirror of episode_dir holding one .pt latent per .mp4")
    parser.add_argument("--vae_weight_path", type=str, required=True,
                        help="Path to the WAN VAE checkpoint (.pth)")
    parser.add_argument("--video_length", type=int, default=18,
                        help="Length of videos in seconds")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed_if_available()

    target_num_frames = args.fps * args.video_length + 1
    if rank == 0:
        print(f"Precomputing {args.video_length}s WAN VAE latents at {args.fps} fps")
        print(f"  from: {args.episode_dir}")
        print(f"  to:   {args.output_dir}")
        print(f"  Each .mp4 should have {target_num_frames} frames "
              f"at {args.height}x{args.width}.")

    vae = get_vae(
        vae_weight_path=args.vae_weight_path,
        device=f"cuda:{local_rank}",
    )

    episodes = sorted(
        d for d in os.listdir(args.episode_dir)
        if osp.isdir(osp.join(args.episode_dir, d))
    )
    episodes_for_this_rank = episodes[rank::world_size]

    for episode in episodes_for_this_rank:
        precompute_episode(
            videos_dir=osp.join(args.episode_dir, episode),
            save_dir=osp.join(args.output_dir, episode),
            vae=vae,
            target_num_frames=target_num_frames,
            fps=args.fps,
            height=args.height,
            width=args.width,
        )
        print(f"[rank {rank}] Done processing {episode}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
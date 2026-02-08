"""
End-to-End Test-Time Training (E2E-TTT) for long-context video generation.

This package is a reference implementation following the design spec in
`e2e_ttt_video_generation.md`.
"""

from .config import (
    InnerLoopConfig,
    OuterLoopConfig,
    InferenceConfig,
    E2ETTTConfig,
)

__all__ = [
    "InnerLoopConfig",
    "OuterLoopConfig",
    "InferenceConfig",
    "E2ETTTConfig",
]


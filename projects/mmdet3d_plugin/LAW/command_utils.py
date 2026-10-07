"""High-level command generation compatible with the official VAD converter."""

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def command_from_future_positions(
    future_positions: ArrayLike,
    lateral_threshold: float = 2.0,
) -> ArrayLike:
    """Generate [right, left, straight] from future ego positions.

    ``future_positions`` must contain cumulative positions relative to the
    current ego frame, not per-step offsets. The official VAD nuScenes
    converter uses the final x coordinate and a 2 m threshold:

      x_final >= +2 m -> right  [1, 0, 0]
      x_final <= -2 m -> left   [0, 1, 0]
      otherwise       -> straight [0, 0, 1]

    The function preserves NumPy/Torch type and supports leading batch dims.
    """
    if lateral_threshold <= 0:
        raise ValueError("lateral_threshold must be positive.")
    if future_positions.shape[-1] != 2:
        raise ValueError(
            "future_positions must have shape [..., M, 2], got "
            f"{future_positions.shape}."
        )

    final_x = future_positions[..., -1, 0]
    if isinstance(future_positions, torch.Tensor):
        command = torch.zeros(
            *final_x.shape, 3,
            dtype=future_positions.dtype,
            device=future_positions.device,
        )
        command[..., 0] = (final_x >= lateral_threshold).to(command.dtype)
        command[..., 1] = (final_x <= -lateral_threshold).to(command.dtype)
        straight = (final_x.abs() < lateral_threshold).to(command.dtype)
        command[..., 2] = straight
        return command

    final_x = np.asarray(final_x)
    command = np.zeros(final_x.shape + (3,), dtype=np.float32)
    command[..., 0] = final_x >= lateral_threshold
    command[..., 1] = final_x <= -lateral_threshold
    command[..., 2] = np.abs(final_x) < lateral_threshold
    return command


def command_from_step_offsets(
    future_offsets: ArrayLike,
    lateral_threshold: float = 2.0,
) -> ArrayLike:
    """Generate command when trajectory is stored as per-step displacements."""
    if isinstance(future_offsets, torch.Tensor):
        positions = future_offsets.cumsum(dim=-2)
    else:
        positions = np.asarray(future_offsets).cumsum(axis=-2)
    return command_from_future_positions(positions, lateral_threshold)

"""Trainable recurrent DreamLite episode model."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from vision_memory.dreamlite.recurrent import DreamLiteRecurrentUpdater


class DreamLiteEpisodeModel(nn.Module):
    def __init__(
        self,
        *,
        pipeline: Any,
        initial_state: Tensor,
        global_seed: int,
        checkpoint_unet: bool,
        learn_initial_state: bool = False,
    ) -> None:
        super().__init__()
        self.updater = DreamLiteRecurrentUpdater(
            pipeline=pipeline,
            global_seed=global_seed,
            checkpoint_unet=checkpoint_unet,
        )
        if learn_initial_state:
            self.initial_state = nn.Parameter(initial_state.detach().clone())
        else:
            self.register_buffer("initial_state", initial_state.detach().clone(), persistent=False)

    def reset_state(self) -> Tensor:
        return self.initial_state.clone()

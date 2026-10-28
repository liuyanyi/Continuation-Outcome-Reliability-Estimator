"""Nemotron-H embedding-mode support for SGLang.

SGLang 0.5.9 ships a NemotronHForCausalLM implementation for generation, but
its forward method does not accept get_embedding. The CORE probe encode path
serves the causal LM with --is-embedding and needs raw last-token hidden states,
so this external package overrides the same architecture name without editing
site-packages.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.nemotron_h import (
    NemotronHForCausalLM as SGLangNemotronHForCausalLM,
)


class NemotronHForCausalLM(SGLangNemotronHForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model.forward(
            input_ids, positions, forward_batch, pp_proxy_tensors, input_embeds
        )
        if self.pp_group.is_last_rank:
            if get_embedding:
                return self.pooler(hidden_states, forward_batch)
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return hidden_states


EntryClass = [NemotronHForCausalLM]

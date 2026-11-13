import math
import os
import inspect
import importlib
from typing import Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.logging import get_logger

logger = get_logger(__name__)


def _build_backbone_from_config(config):
    if getattr(config, "model_type", None) == "nemotron_h":
        config_module = type(config).__module__
        package_name = config_module.rsplit(".", 1)[0]
        modeling_module = importlib.import_module(f"{package_name}.modeling_nemotron_h")
        model_cls = getattr(modeling_module, "NemotronHModel")
        return model_cls(config)

    return AutoModel.from_config(config, trust_remote_code=True)


def _build_loss_weights(
    N: torch.Tensor,
    *,
    weight_mode: str,
) -> torch.Tensor:
    if weight_mode == "none":
        weight = torch.ones_like(N)
    elif weight_mode == "sqrt_n":
        weight = torch.sqrt(torch.clamp(N, min=1e-6))
    elif weight_mode == "n":
        weight = N
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    return weight


def ForProbeLoss(
    pooled_logits: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    label_weights: torch.Tensor | None = None,
    loss_type: str = "soft_bce",
    loss_weight_mode: str = "sqrt_n",
    **kwargs,
) -> torch.Tensor:
    """
    Configurable probe loss over target success rate k / N.

    pooled_logits : (B,)  — value head 输出的原始 logits
    labels        : (B,)  — k，每个节点的正确 rollout 数
    label_weights : (B,)  — N，每个节点的总 rollout 数；为 None 时退化为 N=1
    """
    if pooled_logits is None:
        pooled_logits = kwargs.get("logits")
    if pooled_logits is None:
        raise ValueError("`pooled_logits` or `logits` must be provided")
    if labels is None:
        raise ValueError("`labels` must be provided")

    logits = pooled_logits.squeeze(-1)  # 确保 (B,)
    k = labels.to(logits.device, dtype=logits.dtype)

    if label_weights is not None:
        N = label_weights.to(logits.device, dtype=logits.dtype)
    else:
        N = torch.ones_like(k)

    y = k / N.clamp(min=1e-6)
    weight = _build_loss_weights(N, weight_mode=loss_weight_mode)

    if loss_type == "soft_bce":
        return F.binary_cross_entropy_with_logits(logits, y, weight=weight)

    if loss_type == "mse_prob":
        pred_prob = torch.sigmoid(logits)
        return torch.mean(weight * (pred_prob - y) ** 2)

    raise ValueError(f"Unknown loss_type: {loss_type}")


# This is a generic class for probing that can be used with any transformer-based model.
# 这个类中有大量的self会警告，因为这会是个基类，子类会有这些属性，但在这个类中并没有定义它们。
class GenericForProbing:
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # Similar to `self.model = AutoModel.from_config(config)` but allows to change the base model name if needed in the child class
        setattr(
            self,
            self.base_model_prefix,
            _build_backbone_from_config(config),
        )

        # 具体probing的定义
        self.task_specific_params = config.task_specific_params

        self.prob_num_layers = self.task_specific_params.get("prob_num_layers", 1)
        self.prob_hidden_size = self.task_specific_params.get(
            "prob_hidden_size", config.hidden_size
        )
        self.normalize_hidden_states_before_head = bool(
            self.task_specific_params.get("normalize_hidden_states_before_head", True)
        )
        self.use_hidden_state_feature = bool(
            self.task_specific_params.get("use_hidden_state_feature", True)
        )
        self.use_logit_features = bool(
            self.task_specific_params.get("use_logit_features", False)
        )
        self.use_logit_margin_feature = bool(
            self.task_specific_params.get(
                "use_logit_margin_feature", self.use_logit_features
            )
        )
        self.use_logit_entropy_feature = bool(
            self.task_specific_params.get(
                "use_logit_entropy_feature", self.use_logit_features
            )
        )
        self.logit_feature_detach = bool(
            self.task_specific_params.get("logit_feature_detach", True)
        )
        self.loss_type = str(self.task_specific_params.get("loss_type", "soft_bce"))
        self.loss_weight_mode = str(
            self.task_specific_params.get("loss_weight_mode", "sqrt_n")
        )
        self.logit_feature_dim = int(self.use_logit_margin_feature) + int(
            self.use_logit_entropy_feature
        )
        self.hidden_feature_dim = (
            int(config.hidden_size) if self.use_hidden_state_feature else 0
        )
        if self.loss_type not in {"soft_bce", "mse_prob"}:
            raise ValueError(
                f"Unsupported loss_type={self.loss_type}. Expected one of: soft_bce, mse_prob"
            )
        if self.loss_weight_mode not in {"none", "sqrt_n", "n"}:
            raise ValueError(
                "Unsupported loss_weight_mode="
                f"{self.loss_weight_mode}. Expected one of: none, sqrt_n, n"
            )
        if self.hidden_feature_dim + self.logit_feature_dim <= 0:
            raise ValueError(
                "At least one probe input feature must be enabled. "
                "Set `use_hidden_state_feature=true` or enable a logit feature."
            )
        self.pooling_mode = str(
            self.task_specific_params.get("pooling_mode", "last_token")
        )
        if self.pooling_mode not in {"last_token", "segment_mean"}:
            raise ValueError(
                f"Unsupported pooling_mode={self.pooling_mode}. Expected one of: last_token, segment_mean"
            )
        self.segment_mean_window_tokens = int(
            self.task_specific_params.get("segment_mean_window_tokens", 100)
        )
        if self.segment_mean_window_tokens <= 0:
            raise ValueError("segment_mean_window_tokens must be > 0")

        score_input_dim = self.hidden_feature_dim + self.logit_feature_dim

        # 构建 probe MLP：prob_num_layers=1 为单线性层，>1 时为带 GELU 激活的多层 MLP
        if self.prob_num_layers == 1:
            self.score = nn.Linear(score_input_dim, 1, bias=False)
        else:
            probe_layers: list[nn.Module] = [
                nn.Linear(score_input_dim, self.prob_hidden_size),
                nn.GELU(),
            ]
            for _ in range(self.prob_num_layers - 2):
                probe_layers += [
                    nn.Linear(self.prob_hidden_size, self.prob_hidden_size),
                    nn.GELU(),
                ]
            probe_layers.append(nn.Linear(self.prob_hidden_size, 1, bias=False))
            self.score = nn.Sequential(*probe_layers)

        # 定义 loss function
        self.loss_function = ForProbeLoss

        # Initialize weights and apply final processing
        self.post_init()

    def _compute_logit_features(self, token_hidden: torch.Tensor) -> torch.Tensor:
        """Compute selected logit-derived auxiliary probe features."""
        if self.logit_feature_dim == 0:
            return token_hidden.new_zeros((token_hidden.shape[0], 0))

        base_model = getattr(self, self.base_model_prefix)
        embedding_layer = base_model.get_input_embeddings()
        if embedding_layer is None or not hasattr(embedding_layer, "weight"):
            logger.warning_once(
                "`use_logit_features=true` but input embeddings are unavailable; using zeros."
            )
            return token_hidden.new_zeros(
                (token_hidden.shape[0], self.logit_feature_dim)
            )

        lm_weight = embedding_layer.weight

        # Avoid storing an expensive backward graph through vocab projection.
        with torch.no_grad() if self.logit_feature_detach else torch.enable_grad():
            vocab_logits = F.linear(token_hidden, lm_weight)
            vocab_logits_fp32 = vocab_logits.float()
            probs = torch.softmax(vocab_logits_fp32, dim=-1)
            features_list: list[torch.Tensor] = []

            if self.use_logit_margin_feature:
                top2_values = torch.topk(vocab_logits, k=2, dim=-1).values
                margin = top2_values[:, :1] - top2_values[:, 1:2]
                # Scale to a stable numeric range for head optimization.
                margin_feature = torch.log1p(torch.clamp(margin, min=0.0))
                features_list.append(margin_feature)

            if self.use_logit_entropy_feature:
                entropy = -(probs * torch.log(torch.clamp(probs, min=1e-6))).sum(
                    dim=-1, keepdim=True
                )
                entropy_feature = entropy / max(math.log(vocab_logits.shape[-1]), 1e-6)
                features_list.append(entropy_feature)

            features = torch.cat(features_list, dim=-1)

        return features.to(dtype=token_hidden.dtype)

    def _get_last_non_pad_index(
        self,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.FloatTensor | None,
        batch_size: int,
        device: torch.device,
    ) -> int | torch.Tensor:
        """计算每个样本最后一个非 padding token 的位置索引。"""
        if attention_mask is not None:
            valid_lengths = attention_mask.to(device=device, dtype=torch.int64).sum(
                dim=-1
            )
            return torch.clamp(valid_lengths - 1, min=0)

        if self.config.pad_token_id is None:
            if batch_size != 1:
                logger.warning_once(
                    "No `pad_token_id` and no `attention_mask`; fallback to last token for each sample."
                )
            return -1

        if input_ids is not None:
            # 同时支持左 padding 和右 padding：将非 pad 位置保留其索引值，pad 位置置 0，取 argmax
            non_pad_mask = (input_ids != self.config.pad_token_id).to(
                device, torch.int32
            )
            token_indices = torch.arange(
                input_ids.shape[-1], device=device, dtype=torch.int32
            )
            return (token_indices * non_pad_mask).argmax(-1)

        # inputs_embeds 模式下无法识别 padding，退化为取最后一个 token
        logger.warning_once(
            f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
            "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
        )
        return -1

    def _score_from_state(
        self,
        pooled_state: torch.Tensor,
    ) -> torch.Tensor:
        """Project pooled hidden states to probe logits."""
        features_list: list[torch.Tensor] = []
        if self.use_hidden_state_feature:
            hidden_feature = pooled_state
            if self.normalize_hidden_states_before_head:
                # Stabilize token/layer scale before probe head.
                hidden_feature = F.layer_norm(
                    hidden_feature, hidden_feature.shape[-1:]
                )
            features_list.append(hidden_feature)
        if self.logit_feature_dim > 0:
            logit_features = self._compute_logit_features(pooled_state)
            features_list.append(logit_features)

        score_input = (
            features_list[0]
            if len(features_list) == 1
            else torch.cat(features_list, dim=-1)
        )

        return self.score(score_input)  # [batch, 1]

    def _pool_and_score(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        last_non_pad_token: int | torch.Tensor,
        batch_size: int,
        segment_start_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool hidden states according to configured mode, then score."""
        sequence_length = hidden_states.shape[1]
        if isinstance(last_non_pad_token, int):
            last_token_index = torch.full(
                (batch_size,),
                sequence_length - 1 if last_non_pad_token < 0 else last_non_pad_token,
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            last_token_index = last_non_pad_token.to(
                device=hidden_states.device, dtype=torch.long
            )

        if self.pooling_mode == "segment_mean":
            # Fixed trailing window pooling for diagnostics: mean over last W valid tokens.
            segment_start = torch.clamp(
                last_token_index - (self.segment_mean_window_tokens - 1),
                min=0,
            )

            token_positions = torch.arange(
                sequence_length, device=hidden_states.device
            ).unsqueeze(0)
            segment_mask = (token_positions >= segment_start.unsqueeze(1)) & (
                token_positions <= last_token_index.unsqueeze(1)
            )
            if attention_mask is not None:
                segment_mask = segment_mask & attention_mask.to(
                    device=hidden_states.device, dtype=torch.bool
                )

            denom = torch.clamp(segment_mask.sum(dim=1, keepdim=True), min=1)
            pooled_hidden = (hidden_states * segment_mask.unsqueeze(-1)).sum(
                dim=1
            ) / denom
            return self._score_from_state(pooled_hidden)

        last_hidden = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), last_token_index
        ]  # [batch, hidden_size]
        return self._score_from_state(last_hidden)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        label_weights: torch.FloatTensor | None = None,
        segment_start_positions: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        transformer_outputs: BaseModelOutputWithPast = getattr(
            self, self.base_model_prefix
        )(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state

        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )
        last_non_pad_token = self._get_last_non_pad_index(
            input_ids, attention_mask, inputs_embeds, batch_size, hidden_states.device
        )
        pooled_logits = self._pool_and_score(
            hidden_states,
            attention_mask,
            last_non_pad_token,
            batch_size,
            segment_start_positions=segment_start_positions,
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=pooled_logits,
                labels=labels,
                label_weights=label_weights,
                pooled_logits=pooled_logits,
                loss_type=self.loss_type,
                loss_weight_mode=self.loss_weight_mode,
                config=self.config,
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=getattr(transformer_outputs, "past_key_values", None),
            hidden_states=getattr(transformer_outputs, "hidden_states", None),
            attentions=getattr(transformer_outputs, "attentions", None),
        )

    def export_score_head(self, save_directory: str):
        """导出 probe 的 score 层权重，供后续分析使用。"""
        os.makedirs(save_directory, exist_ok=True)

        # 保存 config，便于后续按同一配置重建 SingleProbeHead。
        self.config.save_pretrained(save_directory)

        single_head = SingleProbeHead(self.config)
        single_head.score.load_state_dict(self.score.state_dict())

        state_dict = {
            k: v.detach().cpu().contiguous() for k, v in single_head.state_dict().items()
        }
        save_file(state_dict, os.path.join(save_directory, "model.safetensors"))


class Qwen3ForProbing(GenericForProbing, Qwen3PreTrainedModel):
    pass


class NemotronHForProbing(GenericForProbing, PreTrainedModel):
    base_model_prefix = "backbone"


SUPPORTED_MODEL_TYPE = {
    "qwen3": Qwen3ForProbing,
    "nemotron_h": NemotronHForProbing,
}


class SingleProbeHead(nn.Module):
    """仅包含probe head的模型定义，用于分离部署"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_labels = getattr(config, "num_labels", 1)
        self.task_specific_params = getattr(config, "task_specific_params", {}) or {}

        self.prob_num_layers = int(self.task_specific_params.get("prob_num_layers", 1))
        self.prob_hidden_size = int(
            self.task_specific_params.get("prob_hidden_size", config.hidden_size)
        )
        self.normalize_hidden_states_before_head = bool(
            self.task_specific_params.get("normalize_hidden_states_before_head", True)
        )
        self.use_hidden_state_feature = bool(
            self.task_specific_params.get("use_hidden_state_feature", True)
        )
        self.use_logit_features = bool(
            self.task_specific_params.get("use_logit_features", False)
        )
        self.use_logit_margin_feature = bool(
            self.task_specific_params.get(
                "use_logit_margin_feature", self.use_logit_features
            )
        )
        self.use_logit_entropy_feature = bool(
            self.task_specific_params.get(
                "use_logit_entropy_feature", self.use_logit_features
            )
        )
        self.logit_feature_detach = bool(
            self.task_specific_params.get("logit_feature_detach", True)
        )
        self.loss_type = str(self.task_specific_params.get("loss_type", "soft_bce"))
        self.loss_weight_mode = str(
            self.task_specific_params.get("loss_weight_mode", "sqrt_n")
        )
        self.logit_feature_dim = int(self.use_logit_margin_feature) + int(
            self.use_logit_entropy_feature
        )
        self.hidden_feature_dim = (
            int(config.hidden_size) if self.use_hidden_state_feature else 0
        )
        if self.loss_type not in {"soft_bce", "mse_prob"}:
            raise ValueError(
                f"Unsupported loss_type={self.loss_type}. Expected one of: soft_bce, mse_prob"
            )
        if self.loss_weight_mode not in {"none", "sqrt_n", "n"}:
            raise ValueError(
                "Unsupported loss_weight_mode="
                f"{self.loss_weight_mode}. Expected one of: none, sqrt_n, n"
            )
        if self.hidden_feature_dim + self.logit_feature_dim <= 0:
            raise ValueError(
                "At least one probe input feature must be enabled. "
                "Set `use_hidden_state_feature=true` or enable a logit feature."
            )

        self.pooling_mode = str(self.task_specific_params.get("pooling_mode", "last_token"))
        if self.pooling_mode not in {"last_token", "segment_mean"}:
            raise ValueError(
                f"Unsupported pooling_mode={self.pooling_mode}. Expected one of: last_token, segment_mean"
            )

        self.segment_mean_window_tokens = int(
            self.task_specific_params.get("segment_mean_window_tokens", 100)
        )
        if self.segment_mean_window_tokens <= 0:
            raise ValueError("segment_mean_window_tokens must be > 0")

        score_input_dim = self.hidden_feature_dim + self.logit_feature_dim
        if self.prob_num_layers == 1:
            self.score = nn.Linear(score_input_dim, 1, bias=False)
        else:
            probe_layers: list[nn.Module] = [
                nn.Linear(score_input_dim, self.prob_hidden_size),
                nn.GELU(),
            ]
            for _ in range(self.prob_num_layers - 2):
                probe_layers += [
                    nn.Linear(self.prob_hidden_size, self.prob_hidden_size),
                    nn.GELU(),
                ]
            probe_layers.append(nn.Linear(self.prob_hidden_size, 1, bias=False))
            self.score = nn.Sequential(*probe_layers)
        
        # Torch compile support
        self._compiled_forward = None
        self._is_compiled = False

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        compile: bool = False,
        compile_options: dict | None = None,
        trust_remote_code: bool = True,
    ) -> "SingleProbeHead":
        """Load SingleProbeHead from `config.json` and `model.safetensors`.
        
        Args:
            path: Path to the model directory
            device: Device to load model onto
            dtype: Data type for model parameters
            compile: If True, compile the model using torch.compile for faster inference
            compile_options: Dict of options to pass to torch.compile (e.g., {'mode': 'reduce-overhead'})
                            Common options: mode ('default', 'reduce-overhead', 'max-autotune'),
                            dynamic_shapes, fullgraph, etc.
        
        Returns:
            Loaded SingleProbeHead instance
        """
        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        model = cls(config)

        state_path = os.path.join(path, "model.safetensors")
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"`model.safetensors` not found under: {path}"
            )

        from safetensors.torch import load_file

        state_dict = load_file(state_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "SingleProbeHead state_dict mismatch while loading from safetensors. "
                f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
            )

        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)

        model.eval()
        
        if compile:
            model.compile_model(compile_options or {})
        
        return model

    def _pool_window_hidden_states(
        self,
        window_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Pool already-sliced token window hidden states."""
        if window_hidden_states.dim() != 3:
            raise ValueError(
                "`window_hidden_states` must be 3D tensor [batch, window_tokens, hidden_size]."
            )

        window_tokens = window_hidden_states.shape[1]
        if window_tokens != self.segment_mean_window_tokens:
            raise ValueError(
                f"Expected `window_hidden_states.shape[1] == segment_mean_window_tokens ({self.segment_mean_window_tokens}), got {window_tokens}."
            )

        if self.pooling_mode == "segment_mean":
            return window_hidden_states.mean(dim=1)
        return window_hidden_states[:, -1, :]

    def _compute_logit_features(
        self,
        token_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.logit_feature_dim == 0:
            return token_hidden.new_zeros((token_hidden.shape[0], 0))

        if lm_head_weight is None and (
            self.use_logit_entropy_feature or self.use_logit_margin_feature
        ):
            raise ValueError(
                "Logit features require `lm_head_weight` to be provided in forward pass."
            )
        lm_head_weight = lm_head_weight.to(device=token_hidden.device, dtype=token_hidden.dtype)
        with torch.no_grad() if self.logit_feature_detach else torch.enable_grad():
            vocab_logits = F.linear(token_hidden, lm_head_weight)
            vocab_logits_fp32 = vocab_logits.float()
            probs = torch.softmax(vocab_logits_fp32, dim=-1)
            features_list: list[torch.Tensor] = []

            if self.use_logit_margin_feature:
                top2_values = torch.topk(vocab_logits, k=2, dim=-1).values
                margin = top2_values[:, :1] - top2_values[:, 1:2]
                margin_feature = torch.log1p(torch.clamp(margin, min=0.0))
                features_list.append(margin_feature)

            if self.use_logit_entropy_feature:
                entropy = -(probs * torch.log(torch.clamp(probs, min=1e-6))).sum(
                    dim=-1, keepdim=True
                )
                entropy_feature = entropy / max(math.log(vocab_logits.shape[-1]), 1e-6)
                features_list.append(entropy_feature)

            features = torch.cat(features_list, dim=-1)

        return features.to(dtype=token_hidden.dtype)

    def _score_from_state(
        self,
        pooled_state: torch.Tensor,
        lm_head_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        features_list: list[torch.Tensor] = []
        if self.use_hidden_state_feature:
            hidden_feature = pooled_state
            if self.normalize_hidden_states_before_head:
                hidden_feature = F.layer_norm(
                    hidden_feature, hidden_feature.shape[-1:]
                )
            features_list.append(hidden_feature)
        if self.logit_feature_dim > 0:
            logit_features = self._compute_logit_features(pooled_state, lm_head_weight)
            features_list.append(logit_features)

        score_input = (
            features_list[0]
            if len(features_list) == 1
            else torch.cat(features_list, dim=-1)
        )

        logits = self.score(score_input)
        # sigmoid
        return F.sigmoid(logits)

    def _forward_impl(
        self,
        window_hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Internal forward implementation for torch.compile."""
        pooled_hidden = self._pool_window_hidden_states(window_hidden_states)
        return self._score_from_state(pooled_hidden, lm_head_weight)

    def forward(
        self,
        window_hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            window_hidden_states: [B, W, H]，已切片窗口的 hidden states。
                其中 W 必须等于 `segment_mean_window_tokens`。
            lm_head_weight: [V, H]，可选，仅在启用 logit features 时需要。

        Returns:
            logits: [B, 1]
        """
        if self._is_compiled and self._compiled_forward is not None:
            return self._compiled_forward(window_hidden_states, lm_head_weight)
        return self._forward_impl(window_hidden_states, lm_head_weight)

    def compile_model(
        self,
        compile_options: dict | None = None,
    ) -> None:
        """Compile the model using torch.compile for faster inference.
        
        Args:
            compile_options: Dict of options to pass to torch.compile.
                            Common options:
                            - mode: 'default', 'reduce-overhead', 'max-autotune'
                            - dynamic_shapes: bool, whether to handle dynamic batch sizes
                            - fullgraph: bool, whether to compile the full graph
                            - backend: 'eager', 'aot_eager', 'aot_cudagraph', etc.
        
        Example:
            model.compile_model({'mode': 'reduce-overhead', 'dynamic_shapes': True})
        """
        if compile_options is None:
            compile_options = {}
        
        try:
            # Check PyTorch version
            if not hasattr(torch, 'compile'):
                logger.warning(
                    "torch.compile not available in this PyTorch version. "
                    "Please upgrade to PyTorch 2.0+. Skipping compilation."
                )
                return
            
            compile_signature = inspect.signature(torch.compile)
            compile_param_names = set(compile_signature.parameters.keys())

            # Build kwargs in a version-compatible way.
            compile_kwargs = {
                "mode": compile_options.get("mode", "reduce-overhead"),
            }

            # Support both naming styles across torch versions:
            # - newer: dynamic
            # - some builds/docs: dynamic_shapes
            # - older/minimal builds: neither (skip dynamic control)
            dynamic_requested = compile_options.get("dynamic_shapes", True)
            if "dynamic" in compile_param_names:
                compile_kwargs["dynamic"] = dynamic_requested
            elif "dynamic_shapes" in compile_param_names:
                compile_kwargs["dynamic_shapes"] = dynamic_requested
            
            # Add optional parameters if provided
            for key in [
                "fullgraph",
                "backend",
                "disable",
                "inline_mass_threshold",
                "decomposition_table",
            ]:
                if key in compile_options:
                    compile_kwargs[key] = compile_options[key]

            # Keep only kwargs supported by the current torch.compile signature.
            compile_kwargs = {
                key: value
                for key, value in compile_kwargs.items()
                if key in compile_param_names
            }
            
            self._compiled_forward = torch.compile(
                self._forward_impl,
                **compile_kwargs
            )
            self._is_compiled = True
            logger.info(
                "Model compiled successfully with options: %s", compile_kwargs
            )
        except Exception as e:
            logger.warning(
                "Failed to compile model: %s. Falling back to eager execution. "
                "This may affect performance.", e
            )
            self._is_compiled = False
            self._compiled_forward = None
    
    def is_compiled(self) -> bool:
        """Check if the model has been compiled."""
        return self._is_compiled


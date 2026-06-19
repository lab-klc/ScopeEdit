import random
from contextlib import contextmanager
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def local_seed_scope(seed: Optional[int]):
    if seed is None:
        yield
        return

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class LoRALinear(nn.Module):
    more_target_lens = None
    more_use_masked_z = True
    more_batch_size = None
    more_bridge_text_mask = None
    more_bridge_image_mask = None
    more_bridge_text_reference = None
    more_bridge_text_reference_mask = None
    more_bridge_visual_anchor_enable = False
    more_bridge_visual_anchor_start = None
    more_bridge_visual_anchor_end = None
    more_bridge_visual_anchor_cache = None

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        lora_alpha: float,
        lora_dropout: float,
        more_nonlinear: str = "none",
        more_layer_norm: bool = False,
        more_layer_norm_eps: float = 1e-5,
        bridge_enable: Optional[bool] = None,
        bridge_rank_private: Optional[int] = None,
        bridge_rank_shared: Optional[int] = None,
        bridge_modality: Optional[str] = None,
        bridge_pooling: Optional[str] = None,
        bridge_scope_separation: Optional[bool] = None,
        scopeedit_enable: Optional[bool] = None,
        scopeedit_rank_private: Optional[int] = None,
        scopeedit_rank_shared: Optional[int] = None,
        scopeedit_modality: Optional[str] = None,
        scopeedit_pooling: Optional[str] = None,
        scopeedit_scope_separation: Optional[bool] = None,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")

        if scopeedit_enable is None:
            scopeedit_enable = bool(bridge_enable) if bridge_enable is not None else False
        if scopeedit_rank_private is None:
            scopeedit_rank_private = bridge_rank_private
        if scopeedit_rank_shared is None:
            scopeedit_rank_shared = 0 if bridge_rank_shared is None else bridge_rank_shared
        if scopeedit_modality is None:
            scopeedit_modality = "text" if bridge_modality is None else bridge_modality
        if scopeedit_pooling is None:
            scopeedit_pooling = "mean" if bridge_pooling is None else bridge_pooling
        if scopeedit_scope_separation is None:
            scopeedit_scope_separation = True if bridge_scope_separation is None else bridge_scope_separation

        self.base_layer = base_layer
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, rank)
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.more_nonlinear = (more_nonlinear or "none").lower()
        self.more_activation = self._resolve_activation(self.more_nonlinear)
        self.bridge_enabled = bool(scopeedit_enable)
        self.scopeedit_enabled = self.bridge_enabled
        self.bridge_modality = str(scopeedit_modality or "text").lower()
        self.scopeedit_modality = self.bridge_modality
        self.bridge_pooling = str(scopeedit_pooling or "mean").lower()
        self.scopeedit_pooling = self.bridge_pooling
        self.bridge_scope_separation = bool(scopeedit_scope_separation)
        self.scopeedit_scope_separation = self.bridge_scope_separation
        if self.bridge_pooling not in ("mean",):
            raise ValueError(f"Unsupported ScopeEdit pooling: {self.bridge_pooling}")

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        if self.bridge_enabled:
            if scopeedit_rank_private is None:
                scopeedit_rank_private = rank - int(scopeedit_rank_shared)
            self.private_rank = int(scopeedit_rank_private)
            self.shared_rank = int(scopeedit_rank_shared)
            if self.private_rank <= 0 or self.shared_rank <= 0:
                raise ValueError(
                    f"ScopeEdit LoRA requires positive private/shared ranks, got "
                    f"private={self.private_rank}, shared={self.shared_rank}"
                )
            if self.private_rank + self.shared_rank != int(rank):
                raise ValueError(
                    f"ScopeEdit LoRA expects private+shared == total rank, got "
                    f"{self.private_rank}+{self.shared_rank}!={rank}"
                )

            self.lora_A_priv = nn.Linear(in_features, self.private_rank, bias=False)
            self.lora_B_priv = nn.Linear(self.private_rank, out_features, bias=False)
            self.lora_A_shr = nn.Linear(in_features, self.shared_rank, bias=False)
            self.lora_B_shr = nn.Linear(self.shared_rank, out_features, bias=False)

            self._move_linear_to_base(self.lora_A_priv)
            self._move_linear_to_base(self.lora_B_priv)
            self._move_linear_to_base(self.lora_A_shr)
            self._move_linear_to_base(self.lora_B_shr)

            self.more_ln_priv = None
            self.more_ln_shr = None
            if more_layer_norm:
                self.more_ln_priv = nn.LayerNorm(
                    self.private_rank, elementwise_affine=False, eps=more_layer_norm_eps
                )
                self.more_ln_shr = nn.LayerNorm(
                    self.shared_rank, elementwise_affine=False, eps=more_layer_norm_eps
                )
                self.more_ln_priv.to(dtype=base_layer.weight.dtype, device=base_layer.weight.device)
                self.more_ln_shr.to(dtype=base_layer.weight.dtype, device=base_layer.weight.device)

            self.register_buffer(
                "more_P_priv",
                torch.eye(self.private_rank, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "more_P_shr",
                torch.eye(self.shared_rank, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "more_last_priv_z",
                torch.zeros(self.private_rank, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "more_last_self_u",
                torch.zeros(self.shared_rank, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "more_last_other_u",
                torch.zeros(self.shared_rank, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "more_last_q",
                torch.zeros(self.shared_rank, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "bridge_other_shared_basis",
                torch.empty(self.shared_rank, 0, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "more_bridge_strength",
                torch.tensor(0.0, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "more_last_alpha",
                torch.tensor(1.0, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "more_last_gamma",
                torch.tensor(0.0, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.more_collect_bridge_basis = False
            self.more_bridge_basis_max_samples = 0
            self.more_bridge_private_cache: List[torch.Tensor] = []
            self.more_bridge_self_cache: List[torch.Tensor] = []
            self.more_bridge_other_cache: List[torch.Tensor] = []
            self.more_bridge_active = False
        else:
            self.private_rank = rank
            self.shared_rank = 0
            self.more_ln = None
            if more_layer_norm:
                self.more_ln = nn.LayerNorm(rank, elementwise_affine=False, eps=more_layer_norm_eps)
                self.more_ln.to(dtype=base_layer.weight.dtype, device=base_layer.weight.device)

            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            self._move_linear_to_base(self.lora_A)
            self._move_linear_to_base(self.lora_B)

            self.more_collect_basis = False
            self.more_basis_max_samples = 0
            self.more_basis_cache = []
            self.register_buffer(
                "more_P",
                torch.eye(rank, device=base_layer.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "more_last_z",
                torch.zeros(rank, device=base_layer.weight.device, dtype=torch.float32),
            )

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.more_group = None
        self.more_name = None

        if self.bridge_enabled:
            self._zero_lora_A()
        else:
            self.reset_lora_A("orthogonal")
        self._zero_lora_B()

    @classmethod
    def stash_scopeedit_context(cls):
        return (
            cls.more_bridge_text_mask,
            cls.more_bridge_image_mask,
            cls.more_bridge_text_reference,
            cls.more_bridge_text_reference_mask,
            cls.more_bridge_visual_anchor_enable,
            cls.more_bridge_visual_anchor_start,
            cls.more_bridge_visual_anchor_end,
            cls.more_bridge_visual_anchor_cache,
        )

    @classmethod
    def stash_bridge_context(cls):
        return cls.stash_scopeedit_context()

    @classmethod
    def set_scopeedit_context(
        cls,
        *,
        text_mask=None,
        image_mask=None,
        text_reference=None,
        text_reference_mask=None,
        visual_anchor_enable=False,
        visual_anchor_start=None,
        visual_anchor_end=None,
    ):
        cls.more_bridge_text_mask = text_mask
        cls.more_bridge_image_mask = image_mask
        cls.more_bridge_text_reference = text_reference
        cls.more_bridge_text_reference_mask = text_reference_mask
        cls.more_bridge_visual_anchor_enable = bool(visual_anchor_enable)
        cls.more_bridge_visual_anchor_start = visual_anchor_start
        cls.more_bridge_visual_anchor_end = visual_anchor_end
        cls.more_bridge_visual_anchor_cache = {} if visual_anchor_enable else None

    @classmethod
    def set_bridge_context(cls, **kwargs):
        cls.set_scopeedit_context(**kwargs)

    @classmethod
    def restore_scopeedit_context(cls, state):
        (
            cls.more_bridge_text_mask,
            cls.more_bridge_image_mask,
            cls.more_bridge_text_reference,
            cls.more_bridge_text_reference_mask,
            cls.more_bridge_visual_anchor_enable,
            cls.more_bridge_visual_anchor_start,
            cls.more_bridge_visual_anchor_end,
            cls.more_bridge_visual_anchor_cache,
        ) = state

    @classmethod
    def restore_bridge_context(cls, state):
        cls.restore_scopeedit_context(state)

    @staticmethod
    def _resolve_activation(name: str):
        if name in ("none", "", None):
            return None
        if name == "gelu":
            return F.gelu
        if name == "relu":
            return F.relu
        if name == "tanh":
            return torch.tanh
        if name in ("silu", "swish"):
            return F.silu
        raise ValueError(f"Unsupported LoRA nonlinearity: {name}")

    def _move_linear_to_base(self, layer: nn.Linear):
        layer.to(dtype=self.base_layer.weight.dtype, device=self.base_layer.weight.device)

    def _zero_lora_B(self):
        if self.bridge_enabled:
            nn.init.zeros_(self.lora_B_priv.weight)
            nn.init.zeros_(self.lora_B_shr.weight)
        else:
            nn.init.zeros_(self.lora_B.weight)

    def _zero_lora_A(self):
        if self.bridge_enabled:
            nn.init.zeros_(self.lora_A_priv.weight)
            nn.init.zeros_(self.lora_A_shr.weight)
        else:
            nn.init.zeros_(self.lora_A.weight)

    def _copy_weight_from_float(self, layer: nn.Linear, weight: torch.Tensor):
        with torch.no_grad():
            layer.weight.copy_(weight.to(dtype=layer.weight.dtype, device=layer.weight.device))

    def _orthogonal_weight(self, rows: int, cols: int):
        tmp = torch.empty(rows, cols, device=self.base_layer.weight.device, dtype=torch.float32)
        nn.init.orthogonal_(tmp)
        return tmp

    def _random_weight(self, rows: int, cols: int, method: str):
        method = (method or "orthogonal").lower()
        if method in ("orthogonal", "default"):
            return self._orthogonal_weight(rows, cols)
        if method == "gaussian":
            weight = torch.empty(rows, cols, device=self.base_layer.weight.device, dtype=torch.float32)
            nn.init.normal_(weight, mean=0.0, std=1.0 / math.sqrt(cols))
            return weight
        if method == "xavier":
            weight = torch.empty(rows, cols, device=self.base_layer.weight.device, dtype=torch.float32)
            nn.init.xavier_normal_(weight)
            return weight
        raise ValueError(f"Unsupported LoRA basis init: {method}")

    def _orthonormalize_rows(
        self,
        rows: Optional[torch.Tensor],
        target_rank: int,
        dim: int,
        exclude: Optional[torch.Tensor] = None,
    ):
        device = self.base_layer.weight.device
        base_vectors: List[torch.Tensor] = []
        if exclude is not None and exclude.numel() > 0:
            for row in exclude.to(device=device, dtype=torch.float32):
                norm = row.norm(p=2)
                if norm > 1e-6:
                    base_vectors.append(row / norm)

        accepted: List[torch.Tensor] = []
        if rows is not None and rows.numel() > 0:
            for row in rows.to(device=device, dtype=torch.float32):
                vec = row.clone()
                for basis in base_vectors:
                    vec = vec - torch.dot(vec, basis) * basis
                for basis in accepted:
                    vec = vec - torch.dot(vec, basis) * basis
                norm = vec.norm(p=2)
                if norm > 1e-6:
                    accepted.append(vec / norm)
                if len(accepted) >= target_rank:
                    break

        while len(accepted) < target_rank:
            vec = torch.randn(dim, device=device, dtype=torch.float32)
            for basis in base_vectors:
                vec = vec - torch.dot(vec, basis) * basis
            for basis in accepted:
                vec = vec - torch.dot(vec, basis) * basis
            norm = vec.norm(p=2)
            if norm > 1e-6:
                accepted.append(vec / norm)

        if target_rank == 0:
            return torch.empty(0, dim, device=device, dtype=torch.float32)
        return torch.stack(accepted[:target_rank], dim=0)

    def _apply_branch_transform(self, ax: torch.Tensor, branch: str):
        if self.more_activation is not None:
            ax = self.more_activation(ax)
        if not self.bridge_enabled:
            if self.more_ln is not None:
                ax = self.more_ln(ax)
            return ax

        if branch == "priv" and self.more_ln_priv is not None:
            ax = self.more_ln_priv(ax)
        if branch == "shr" and self.more_ln_shr is not None:
            ax = self.more_ln_shr(ax)
        return ax

    def reset_lora_A(self, method: str = "orthogonal"):
        method = (method or "orthogonal").lower()
        if method == "default":
            method = "orthogonal"
        if self.bridge_enabled:
            if method == "orthogonal" and self.rank > self.base_layer.in_features:
                raise ValueError(
                    f"ScopeEdit LoRA requires rank <= in_features for orthogonal basis init, got "
                    f"rank={self.rank}, in_features={self.base_layer.in_features}"
                )
            full_basis = self._random_weight(
                self.rank,
                self.base_layer.in_features,
                method,
            )
            if self.bridge_scope_separation:
                shared_basis = full_basis[: self.shared_rank]
                private_basis = full_basis[self.shared_rank : self.shared_rank + self.private_rank]
            else:
                if self.shared_rank > self.private_rank:
                    raise ValueError(
                        "scopeedit_scope_separation=False requires rank_shared <= rank_private "
                        f"so the shared branch can reuse the private row space, got "
                        f"rank_shared={self.shared_rank}, rank_private={self.private_rank}"
                    )
                private_basis = full_basis[: self.private_rank]
                shared_basis = private_basis[: self.shared_rank].clone()
            self._copy_weight_from_float(
                self.lora_A_priv,
                private_basis,
            )
            self._copy_weight_from_float(
                self.lora_A_shr,
                shared_basis,
            )
            if self.bridge_modality == "text":
                self.bridge_other_shared_basis = shared_basis.detach().clone().to(
                    device=self.base_layer.weight.device,
                    dtype=torch.float32,
                )
            else:
                self.bridge_other_shared_basis = torch.empty(
                    self.shared_rank,
                    0,
                    device=self.base_layer.weight.device,
                    dtype=torch.float32,
                )
            self.more_bridge_strength.fill_(1.0)
            self.reset_more_stats()
            return

        self._copy_weight_from_float(
            self.lora_A, self._random_weight(self.rank, self.base_layer.in_features, method)
        )
        self.reset_more_stats()

    def reset_more_stats(self):
        if self.bridge_enabled:
            self.more_P_priv.copy_(
                torch.eye(self.private_rank, device=self.more_P_priv.device, dtype=self.more_P_priv.dtype)
            )
            self.more_P_shr.copy_(
                torch.eye(self.shared_rank, device=self.more_P_shr.device, dtype=self.more_P_shr.dtype)
            )
            self.more_last_priv_z.zero_()
            self.more_last_self_u.zero_()
            self.more_last_other_u.zero_()
            self.more_last_q.zero_()
            self.more_last_alpha.fill_(1.0)
            self.more_last_gamma.zero_()
            return

        self.more_P.copy_(torch.eye(self.rank, device=self.more_P.device, dtype=self.more_P.dtype))
        self.more_last_z.zero_()

    def reset_basis_cache(self, max_samples: int):
        if self.bridge_enabled:
            self.more_bridge_basis_max_samples = max(0, int(max_samples))
            self.more_bridge_private_cache = []
            self.more_bridge_self_cache = []
            self.more_bridge_other_cache = []
            return
        self.more_basis_max_samples = max(0, int(max_samples))
        self.more_basis_cache = []

    def clear_basis_cache(self):
        if self.bridge_enabled:
            self.more_collect_bridge_basis = False
            self.more_bridge_basis_max_samples = 0
            self.more_bridge_private_cache = []
            self.more_bridge_self_cache = []
            self.more_bridge_other_cache = []
            return
        self.more_collect_basis = False
        self.more_basis_max_samples = 0
        self.more_basis_cache = []

    def basis_cache_size(self):
        if self.bridge_enabled:
            private_size = sum(chunk.size(0) for chunk in self.more_bridge_private_cache)
            self_size = sum(chunk.size(0) for chunk in self.more_bridge_self_cache)
            return min(private_size, self_size)
        return sum(chunk.size(0) for chunk in self.more_basis_cache)

    def _sequence_view(self, tensor: torch.Tensor):
        if tensor.dim() >= 3:
            return tensor, tensor.size(0), tensor.size(1)
        if tensor.dim() == 2:
            bsz = getattr(LoRALinear, "more_batch_size", None)
            if bsz is not None and bsz > 0 and tensor.size(0) % bsz == 0:
                seq_len = tensor.size(0) // bsz
                return tensor.view(bsz, seq_len, tensor.size(-1)), bsz, seq_len
            return tensor.unsqueeze(0), 1, tensor.size(0)
        if tensor.dim() == 1:
            return tensor.view(1, 1, -1), 1, 1
        return None, None, None

    def _extract_basis_vectors(self, tensor: torch.Tensor):
        if tensor.dim() == 1:
            return tensor.view(1, -1)

        is_text_aligned = not (
            self.more_group == "vision_proj"
            or str(self.more_group).startswith("vision_encoder.")
            or str(self.more_group).startswith("qformer.")
        )
        use_masked = (
            LoRALinear.more_use_masked_z
            and is_text_aligned
            and isinstance(LoRALinear.more_target_lens, (list, tuple))
        )
        if use_masked:
            tensor_view, batch_size, seq_len = self._sequence_view(tensor)
            if tensor_view is not None and len(LoRALinear.more_target_lens) == batch_size:
                mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=tensor.device)
                for i, tlen in enumerate(LoRALinear.more_target_lens):
                    if tlen is None or tlen <= 0:
                        continue
                    start = max(0, seq_len - int(tlen))
                    mask[i, start:seq_len] = True
                if mask.any():
                    return tensor_view[mask].view(-1, tensor_view.size(-1))

        if tensor.dim() >= 3:
            return tensor.reshape(-1, tensor.size(-1))
        if tensor.dim() == 2:
            return tensor
        return None

    def _maybe_collect_basis(self, x: torch.Tensor):
        if self.bridge_enabled or not self.more_collect_basis or self.more_basis_max_samples <= 0:
            return

        remaining = self.more_basis_max_samples - self.basis_cache_size()
        if remaining <= 0:
            return

        vectors = self._extract_basis_vectors(x)
        if vectors is None or vectors.numel() == 0:
            return

        if vectors.size(0) > remaining:
            vectors = vectors[:remaining]

        self.more_basis_cache.append(vectors.detach().to(device="cpu", dtype=torch.float32))

    def _maybe_collect_bridge_private_basis(self, x: torch.Tensor):
        if (
            not self.bridge_enabled
            or not self.more_collect_bridge_basis
            or self.more_bridge_basis_max_samples <= 0
        ):
            return

        private_size = sum(chunk.size(0) for chunk in self.more_bridge_private_cache)
        remaining = self.more_bridge_basis_max_samples - private_size
        if remaining <= 0:
            return

        vectors = self._extract_basis_vectors(x)
        if vectors is None or vectors.numel() == 0:
            return

        if vectors.size(0) > remaining:
            vectors = vectors[:remaining]

        self.more_bridge_private_cache.append(vectors.detach().to(device="cpu", dtype=torch.float32))

    def _masked_pool(self, seq: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is None:
            return None
        if mask.dim() != 2 or mask.shape[:2] != seq.shape[:2]:
            return None
        pooled = []
        for i in range(seq.size(0)):
            cur = mask[i]
            if cur.any():
                pooled.append(seq[i][cur].mean(dim=0))
            else:
                pooled.append(torch.zeros(seq.size(-1), device=seq.device, dtype=seq.dtype))
        return torch.stack(pooled, dim=0)

    def _pool_unmasked(self, tensor: Optional[torch.Tensor]):
        if tensor is None:
            return None
        if tensor.dim() == 3:
            return tensor.mean(dim=1)
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() == 1:
            return tensor.view(1, -1)
        return None

    @staticmethod
    def _group_layer_index(group_name):
        parts = str(group_name).split(".")
        if not parts:
            return None
        try:
            return int(parts[-1])
        except ValueError:
            return None

    def _is_text_aligned_group(self):
        return not (
            self.more_group == "vision_proj"
            or str(self.more_group).startswith("vision_encoder.")
            or str(self.more_group).startswith("qformer.")
        )

    def _can_use_target_mask(self):
        return LoRALinear.more_use_masked_z and isinstance(LoRALinear.more_target_lens, (list, tuple))

    def _build_target_mask(self, seq: Optional[torch.Tensor]):
        if seq is None or seq.dim() != 3 or not self._can_use_target_mask():
            return None

        batch_size, seq_len = seq.size(0), seq.size(1)
        if len(LoRALinear.more_target_lens) != batch_size:
            return None

        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=seq.device)
        for i, tlen in enumerate(LoRALinear.more_target_lens):
            if tlen is None or tlen <= 0:
                continue
            start = max(0, seq_len - int(tlen))
            mask[i, start:seq_len] = True
        return mask

    def _reduce_sequence_code(self, tensor: torch.Tensor, mask: Optional[torch.Tensor] = None):
        seq, _, _ = self._sequence_view(tensor)
        if seq is not None:
            if (
                mask is not None
                and mask.dim() == 2
                and mask.shape[:2] == seq.shape[:2]
                and mask.any()
            ):
                return seq[mask].view(-1, seq.size(-1)).mean(dim=0)
            return seq.mean(dim=(0, 1))

        while tensor.dim() > 1:
            tensor = tensor.mean(dim=0)
        return tensor

    def _update_visual_anchor_cache(self, image_key: Optional[torch.Tensor]):
        if (
            not self.bridge_enabled
            or self.bridge_modality != "text"
            or not LoRALinear.more_bridge_visual_anchor_enable
            or image_key is None
        ):
            return

        cache = LoRALinear.more_bridge_visual_anchor_cache
        if cache is None:
            return
        cache.setdefault(str(self.more_group), []).append(image_key.detach().to(dtype=torch.float32))

    def _resolve_visual_anchor(self, fallback_key: Optional[torch.Tensor]):
        if (
            fallback_key is None
            or not self.bridge_enabled
            or self.bridge_modality != "text"
            or not LoRALinear.more_bridge_visual_anchor_enable
        ):
            return fallback_key

        start = LoRALinear.more_bridge_visual_anchor_start
        end = LoRALinear.more_bridge_visual_anchor_end
        cache = LoRALinear.more_bridge_visual_anchor_cache
        current_idx = self._group_layer_index(self.more_group)
        if start is None or end is None or cache is None or current_idx is None or current_idx <= end:
            return fallback_key

        group_means = []
        for group_name, values in cache.items():
            layer_idx = self._group_layer_index(group_name)
            if layer_idx is None or layer_idx < start or layer_idx > end or layer_idx >= current_idx:
                continue
            if not values:
                continue
            group_means.append(torch.stack(values, dim=0).mean(dim=0))

        if not group_means:
            return fallback_key

        anchor = torch.stack(group_means, dim=0).mean(dim=0)
        return anchor.to(device=fallback_key.device, dtype=fallback_key.dtype)

    def _extract_bridge_keys(self, x: torch.Tensor):
        seq, _, _ = self._sequence_view(x)
        if seq is None:
            return None, None

        if self.bridge_modality == "text":
            text_mask = LoRALinear.more_bridge_text_mask
            image_mask = LoRALinear.more_bridge_image_mask
            target_mask = self._build_target_mask(seq)
            target_text_mask = None
            if target_mask is not None:
                target_text_mask = target_mask if text_mask is None else (target_mask & text_mask)

            self_key = self._masked_pool(seq, target_text_mask) if target_text_mask is not None else None
            if self_key is None:
                self_key = self._masked_pool(seq, text_mask)
            other_key = self._masked_pool(seq, image_mask)
            self._update_visual_anchor_cache(other_key)
            other_key = self._resolve_visual_anchor(other_key)
            if self_key is None:
                self_key = seq.mean(dim=1)
            return self_key, other_key

        self_key = seq.mean(dim=1)
        other_ref = LoRALinear.more_bridge_text_reference
        other_mask = LoRALinear.more_bridge_text_reference_mask
        if other_ref is None:
            return self_key, None

        if isinstance(other_ref, torch.Tensor):
            if other_ref.dim() >= 3:
                other_seq, _, _ = self._sequence_view(other_ref)
                target_other_mask = self._build_target_mask(other_seq)
                effective_other_mask = other_mask
                if target_other_mask is not None:
                    effective_other_mask = (
                        target_other_mask
                        if effective_other_mask is None
                        else (effective_other_mask & target_other_mask)
                    )
                other_key = self._masked_pool(other_seq, effective_other_mask)
                if other_key is None:
                    other_key = self._masked_pool(other_seq, other_mask)
                if other_key is None:
                    other_key = other_seq.mean(dim=1)
            elif other_ref.dim() == 2:
                other_key = other_ref
            else:
                other_key = other_ref.view(1, -1)
            return self_key, other_key

        return self_key, None

    def _maybe_collect_bridge_basis(self, self_key: Optional[torch.Tensor], other_key: Optional[torch.Tensor]):
        if (
            not self.bridge_enabled
            or not self.more_collect_bridge_basis
            or self.more_bridge_basis_max_samples <= 0
            or self_key is None
        ):
            return

        self_size = sum(chunk.size(0) for chunk in self.more_bridge_self_cache)
        remaining = self.more_bridge_basis_max_samples - self_size
        if remaining <= 0:
            return

        if other_key is not None and self_key.size(0) != other_key.size(0):
            size = min(self_key.size(0), other_key.size(0))
            self_key = self_key[:size]
            other_key = other_key[:size]

        if self_key.size(0) > remaining:
            self_key = self_key[:remaining]
            if other_key is not None:
                other_key = other_key[:remaining]

        self.more_bridge_self_cache.append(self_key.detach().to(device="cpu", dtype=torch.float32))
        if other_key is not None:
            self.more_bridge_other_cache.append(other_key.detach().to(device="cpu", dtype=torch.float32))

    def _store_nonbridge_state(self, z: torch.Tensor):
        z_seq, _, _ = self._sequence_view(z)
        mask = self._build_target_mask(z_seq) if self._is_text_aligned_group() else None
        z = self._reduce_sequence_code(z, mask)
        self.more_last_z.copy_(z.detach().to(self.more_last_z.dtype))

    def _store_bridge_state(self, x: torch.Tensor, private_z: torch.Tensor):
        private_seq, _, _ = self._sequence_view(private_z)
        private_mask = self._build_target_mask(private_seq) if self._is_text_aligned_group() else None
        private_code = self._reduce_sequence_code(private_z, private_mask)
        self.more_last_priv_z.copy_(private_code.detach().to(self.more_last_priv_z.dtype))

        self_key, other_key = self._extract_bridge_keys(x)
        self._maybe_collect_bridge_basis(self_key, other_key)
        if self_key is None:
            self.more_last_self_u.zero_()
            self.more_last_other_u.zero_()
            return

        self_key = self_key.float().mean(dim=0)
        shr_basis = self.lora_A_shr.weight.detach().float()
        self.more_last_self_u.copy_(torch.mv(shr_basis, self_key).to(self.more_last_self_u.dtype))

        if (
            other_key is None
            or not isinstance(self.bridge_other_shared_basis, torch.Tensor)
            or self.bridge_other_shared_basis.numel() == 0
        ):
            self.more_last_other_u.zero_()
            return

        other_key = other_key.float().mean(dim=0)
        if self.bridge_other_shared_basis.size(1) != other_key.numel():
            self.more_last_other_u.zero_()
            return
        other_basis = self.bridge_other_shared_basis.detach().float()
        self.more_last_other_u.copy_(torch.mv(other_basis, other_key).to(self.more_last_other_u.dtype))

    def _complete_basis(self, rows: Optional[torch.Tensor], dim: int, rank: int, exclude=None):
        return self._orthonormalize_rows(rows, rank, dim, exclude=exclude)

    def _basis_from_cache(
        self,
        cache: List[torch.Tensor],
        method: str,
        rank: int,
        dim: int,
        exclude: Optional[torch.Tensor] = None,
    ):
        if not cache:
            return None

        samples = torch.cat(cache, dim=0)
        if samples.dim() != 2 or samples.size(0) == 0 or samples.size(1) != dim:
            return None

        work = samples
        if method == "pca":
            work = work - work.mean(dim=0, keepdim=True)

        try:
            _, _, vh = torch.linalg.svd(work, full_matrices=False)
        except RuntimeError:
            return None

        if vh.numel() == 0:
            return None

        basis_rows = vh[: min(rank, vh.size(0))]
        return self._complete_basis(basis_rows, dim, rank, exclude=exclude)

    def initialize_lora_A_from_data(self, method: str):
        if self.bridge_enabled:
            return False

        method = (method or "pca").lower()
        if method not in ("pca", "svd"):
            raise ValueError(f"Data-driven LoRA basis only supports pca/svd, got {method}")
        if not self.more_basis_cache:
            self.reset_lora_A("orthogonal")
            return False

        samples = torch.cat(self.more_basis_cache, dim=0)
        if samples.dim() != 2 or samples.size(0) == 0:
            self.reset_lora_A("orthogonal")
            return False

        work = samples
        if method == "pca":
            work = work - work.mean(dim=0, keepdim=True)

        try:
            _, _, vh = torch.linalg.svd(work, full_matrices=False)
        except RuntimeError:
            self.reset_lora_A("orthogonal")
            return False

        if vh.numel() == 0:
            self.reset_lora_A("orthogonal")
            return False

        basis_rows = vh[: min(self.rank, vh.size(0))]
        full_basis = self._complete_basis(basis_rows, self.base_layer.in_features, self.rank)
        self._copy_weight_from_float(self.lora_A, full_basis)
        self.reset_more_stats()
        return True

    def initialize_scopeedit_basis_from_data(self, method: str):
        if not self.bridge_enabled:
            return False

        method = (method or "pca").lower()
        if method not in ("pca", "svd"):
            raise ValueError(f"Data-driven ScopeEdit LoRA basis only supports pca/svd, got {method}")

        in_features = self.base_layer.in_features
        private_basis = self._basis_from_cache(
            self.more_bridge_private_cache,
            method,
            self.private_rank,
            in_features,
        )
        if private_basis is None:
            self.reset_lora_A("orthogonal")
            return False

        if self.bridge_scope_separation:
            shared_cache = list(self.more_bridge_self_cache)
            if self.more_bridge_other_cache:
                other_dim = self.more_bridge_other_cache[0].size(1)
                if other_dim == in_features:
                    shared_cache = shared_cache + list(self.more_bridge_other_cache)
            shared_basis = self._basis_from_cache(
                shared_cache,
                method,
                self.shared_rank,
                in_features,
                exclude=private_basis,
            )
            if shared_basis is None:
                shared_basis = self._complete_basis(
                    None,
                    in_features,
                    self.shared_rank,
                    exclude=private_basis,
                )
        else:
            if self.shared_rank > self.private_rank:
                raise ValueError(
                    "scopeedit_scope_separation=False requires rank_shared <= rank_private "
                    f"so the shared branch can reuse the private row space, got "
                    f"rank_shared={self.shared_rank}, rank_private={self.private_rank}"
                )
            shared_basis = private_basis[: self.shared_rank].clone()

        self._copy_weight_from_float(self.lora_A_priv, private_basis)
        self._copy_weight_from_float(self.lora_A_shr, shared_basis)
        if self.bridge_modality == "text":
            self.bridge_other_shared_basis = shared_basis.detach().clone().to(
                device=self.base_layer.weight.device,
                dtype=torch.float32,
            )
        else:
            self.bridge_other_shared_basis = torch.empty(
                self.shared_rank,
                0,
                device=self.base_layer.weight.device,
                dtype=torch.float32,
            )
        self.reset_more_stats()
        self.more_bridge_strength.fill_(1.0)
        return True

    def initialize_bridge_basis_from_data(self, method: str):
        return self.initialize_scopeedit_basis_from_data(method)

    def initialize_scopeedit_basis(self, method: str = "orthogonal"):
        if not self.bridge_enabled:
            return False
        self.reset_lora_A(method)
        self.reset_more_stats()
        self.more_bridge_strength.fill_(1.0)
        return True

    def initialize_bridge_basis(self, method: str = "orthogonal"):
        return self.initialize_scopeedit_basis(method)

    def _recursive_p_update(self, p_buf, code, rls_lambda: float, p_old: Optional[torch.Tensor] = None):
        if code is None:
            return False
        z = code.detach().to(dtype=torch.float32)
        if z.dim() != 1:
            z = z.reshape(-1)
        if z.numel() != p_buf.size(0):
            return False

        P_old = p_buf.detach().to(dtype=torch.float32).clone() if p_old is None else p_old
        with torch.no_grad():
            denom = rls_lambda + torch.dot(z, P_old @ z)
            if denom.abs().item() < 1e-12:
                return False
            Pz = P_old @ z
            P_new = P_old - torch.outer(Pz, Pz) / denom
            p_buf.copy_(P_new)
        return True

    def _recursive_update(self, a_layer, b_layer, p_buf, code, eta: float, rls_lambda: float, scale: float):
        gA = a_layer.weight.grad
        gB = b_layer.weight.grad
        if scale <= 0 or (gA is None and gB is None) or code is None:
            return False
        # z_t
        z = code.detach().to(dtype=torch.float32)
        if z.dim() != 1:
            z = z.reshape(-1)
        if z.numel() != p_buf.size(0):
            return False
        # P_{t-1}
        P_old = p_buf.detach().to(dtype=torch.float32).clone()

        with torch.no_grad():
            if gA is not None:
                delta_A = (P_old @ gA.float()) * (eta * scale)
                a_layer.weight.add_(-delta_A.to(dtype=a_layer.weight.dtype))
            if gB is not None:
                delta_B = (gB.float() @ P_old) * (eta * scale)
                b_layer.weight.add_(-delta_B.to(dtype=b_layer.weight.dtype))

        return self._recursive_p_update(p_buf, z, rls_lambda, p_old=P_old)

    def accumulate_current_key(
        self,
        rls_lambda_private: float,
        rls_lambda_shared: Optional[float] = None,
        include_shared: bool = False,
    ):
        if not self.bridge_enabled:
            updated_private = self._recursive_p_update(
                self.more_P,
                self.more_last_z,
                rls_lambda_private,
            )
            return {
                "updated_private": updated_private,
                "updated_shared": False,
            }

        updated_private = self._recursive_p_update(
            self.more_P_priv,
            self.more_last_priv_z,
            rls_lambda_private,
        )
        updated_shared = False
        if include_shared:
            shared_code = self.more_last_q
            if shared_code.float().norm(p=2).item() <= 1e-12:
                shared_code = self.more_last_self_u
            updated_shared = self._recursive_p_update(
                self.more_P_shr,
                shared_code,
                rls_lambda_private if rls_lambda_shared is None else rls_lambda_shared,
            )

        return {
            "updated_private": updated_private,
            "updated_shared": updated_shared,
        }

    def _bridge_gate_values(
        self,
        tau: float,
        beta: float,
        eps: float = 1e-6,
        scope_mode: str = "gated",
    ):
        scope_mode = str(scope_mode or "gated").lower()
        if scope_mode not in ("gated", "always_on"):
            raise ValueError(f"Unsupported scopeedit_scope_mode: {scope_mode}")

        self_u = self.more_last_self_u.float()
        other_u = self.more_last_other_u.float()
        self_norm = self_u.norm(p=2)
        other_norm = other_u.norm(p=2)
        if self_norm.item() <= eps or other_norm.item() <= eps:
            return {
                "alpha": 1.0,
                "gamma_self": 0.0,
                "gamma_other": 0.0,
                "cos": 0.0,
                "support": 0.0,
                "q": None,
            }

        cos = torch.dot(self_u, other_u) / (self_norm * other_norm + eps)
        support = torch.minimum(self_norm, other_norm) / (torch.maximum(self_norm, other_norm) + eps)
        denom = self_norm + other_norm + eps
        mix_self = other_norm / denom
        mix_other = self_norm / denom

        if scope_mode == "always_on":
            shared_scale = torch.tensor(1.0, device=self_u.device, dtype=torch.float32)
            q = mix_other * self_u + mix_self * other_u
            q_norm = q.norm(p=2)
            if q_norm.item() > eps:
                q = q / q_norm
            else:
                q = None
            return {
                "alpha": 1.0,
                "gamma_self": float(shared_scale.item()),
                "gamma_other": float(shared_scale.item()),
                "cos": float(cos.item()),
                "support": float(support.item()),
                "q": q,
            }

        gamma = torch.sigmoid(torch.tensor(beta, device=self_u.device, dtype=torch.float32) * (cos - tau))
        gamma = gamma * support
        gamma_self = gamma * mix_self
        gamma_other = gamma * mix_other
        q = gamma_other * self_u + gamma_self * other_u
        q_norm = q.norm(p=2)
        if q_norm.item() > eps:
            q = q / q_norm
        else:
            q = None

        return {
            "alpha": float((1.0 - gamma_self).item()),
            "gamma_self": float(gamma_self.item()),
            "gamma_other": float(gamma_other.item()),
            "cos": float(cos.item()),
            "support": float(support.item()),
            "q": q,
        }

    def more_score_value(self, score_norm: str = "none"):
        score_norm = (score_norm or "none").lower()
        grads = []
        if self.bridge_enabled:
            # Keep group selection aligned with the original M-ORE private backbone;
            # the shared branch should behave as an auxiliary residual writer.
            grads = [
                self.lora_A_priv.weight.grad,
                self.lora_B_priv.weight.grad,
            ]
            params = [
                self.lora_A_priv.weight,
                self.lora_B_priv.weight,
            ]
        else:
            grads = [self.lora_A.weight.grad, self.lora_B.weight.grad]
            params = [self.lora_A.weight, self.lora_B.weight]

        score = 0.0
        denom = 1.0
        if score_norm == "param":
            denom = math.sqrt(sum(p.numel() for p, g in zip(params, grads) if g is not None) or 1.0)
        for grad in grads:
            if grad is not None:
                score += grad.float().norm(p=2).item()
        return score / denom

    def snapshot_more_state(self, restore_p: bool):
        if self.bridge_enabled:
            return {
                "lora_A_priv": self.lora_A_priv.weight.detach().clone(),
                "lora_B_priv": self.lora_B_priv.weight.detach().clone(),
                "lora_A_shr": self.lora_A_shr.weight.detach().clone(),
                "lora_B_shr": self.lora_B_shr.weight.detach().clone(),
                "more_P_priv": self.more_P_priv.detach().clone() if restore_p else None,
                "more_P_shr": self.more_P_shr.detach().clone() if restore_p else None,
            }
        return {
            "lora_A": self.lora_A.weight.detach().clone(),
            "lora_B": self.lora_B.weight.detach().clone(),
            "more_P": self.more_P.detach().clone() if restore_p else None,
        }

    def restore_more_state(self, state):
        with torch.no_grad():
            if self.bridge_enabled:
                self.lora_A_priv.weight.copy_(state["lora_A_priv"])
                self.lora_B_priv.weight.copy_(state["lora_B_priv"])
                self.lora_A_shr.weight.copy_(state["lora_A_shr"])
                self.lora_B_shr.weight.copy_(state["lora_B_shr"])
                if state.get("more_P_priv") is not None:
                    self.more_P_priv.copy_(state["more_P_priv"])
                if state.get("more_P_shr") is not None:
                    self.more_P_shr.copy_(state["more_P_shr"])
            else:
                self.lora_A.weight.copy_(state["lora_A"])
                self.lora_B.weight.copy_(state["lora_B"])
                if state.get("more_P") is not None:
                    self.more_P.copy_(state["more_P"])

    def apply_online_update(
        self,
        *,
        eta_private: float,
        lambda_private: float,
        eta_shared: Optional[float] = None,
        lambda_shared: Optional[float] = None,
        bridge_tau: Optional[float] = None,
        bridge_beta: Optional[float] = None,
        bridge_scope_mode: Optional[str] = None,
        scopeedit_tau: Optional[float] = None,
        scopeedit_beta: Optional[float] = None,
        scopeedit_scope_mode: Optional[str] = None,
        enable_shared: bool = True,
    ):
        if scopeedit_tau is None:
            scopeedit_tau = 0.0 if bridge_tau is None else bridge_tau
        if scopeedit_beta is None:
            scopeedit_beta = 10.0 if bridge_beta is None else bridge_beta
        if scopeedit_scope_mode is None:
            scopeedit_scope_mode = "gated" if bridge_scope_mode is None else bridge_scope_mode

        if not self.bridge_enabled:
            updated = self._recursive_update(
                self.lora_A,
                self.lora_B,
                self.more_P,
                self.more_last_z,
                eta_private,
                lambda_private,
                1.0,
            )
            return {
                "alpha": 1.0,
                "gamma": 0.0,
                "cos": 0.0,
                "support": 0.0,
                "updated_private": updated,
                "updated_shared": False,
            }

        eta_shared = eta_private if eta_shared is None else eta_shared
        lambda_shared = lambda_private if lambda_shared is None else lambda_shared
        gate = self._bridge_gate_values(
            scopeedit_tau,
            scopeedit_beta,
            scope_mode=scopeedit_scope_mode,
        )
        if not enable_shared or not self.more_bridge_active:
            gate["gamma_self"] = 0.0
            gate["q"] = None

        private_scale = 1.0
        self.more_last_alpha.fill_(private_scale)
        self.more_last_gamma.fill_(gate["gamma_self"])
        if gate["q"] is not None:
            self.more_last_q.copy_(gate["q"].to(dtype=self.more_last_q.dtype))
        else:
            self.more_last_q.zero_()

        updated_private = self._recursive_update(
            self.lora_A_priv,
            self.lora_B_priv,
            self.more_P_priv,
            self.more_last_priv_z,
            eta_private,
            lambda_private,
            private_scale,
        )
        updated_shared = False
        if gate["q"] is not None and gate["gamma_self"] > 0:
            updated_shared = self._recursive_update(
                self.lora_A_shr,
                self.lora_B_shr,
                self.more_P_shr,
                gate["q"],
                eta_shared,
                lambda_shared,
                gate["gamma_self"],
            )

        return {
            "alpha": private_scale,
            "gamma": gate["gamma_self"],
            "cos": gate["cos"],
            "support": gate["support"],
            "updated_private": updated_private,
            "updated_shared": updated_shared,
        }

    def more_update(self, eta: float, rls_lambda: float):
        if self.bridge_enabled:
            self.apply_online_update(
                eta_private=eta,
                lambda_private=rls_lambda,
                eta_shared=eta,
                lambda_shared=rls_lambda,
            )
            return

        self._recursive_update(
            self.lora_A,
            self.lora_B,
            self.more_P,
            self.more_last_z,
            eta,
            rls_lambda,
            1.0,
        )

    def forward(self, x):
        with torch.no_grad():
            self._maybe_collect_basis(x)
            self._maybe_collect_bridge_private_basis(x)

        base_out = self.base_layer(x)

        if self.bridge_enabled:
            if self.more_bridge_strength.item() <= 0:
                raise RuntimeError(
                    "ScopeEdit LoRA basis is not initialized. Run the ScopeEdit initialization "
                    "path before calling forward."
                )
            ax_priv = self.lora_A_priv(self.lora_dropout(x))
            ax_priv = self._apply_branch_transform(ax_priv, "priv")
            bx_priv = self.lora_B_priv(ax_priv)

            ax_shr = self.lora_A_shr(self.lora_dropout(x))
            ax_shr = self._apply_branch_transform(ax_shr, "shr")
            bx_shr = self.lora_B_shr(ax_shr)

            with torch.no_grad():
                self._store_bridge_state(x, ax_priv)
            return base_out + self.scaling * (bx_priv + bx_shr)

        ax = self.lora_A(self.lora_dropout(x))
        ax = self._apply_branch_transform(ax, "base")
        bx = self.lora_B(ax)

        with torch.no_grad():
            self._store_nonbridge_state(ax)

        return base_out + self.scaling * bx

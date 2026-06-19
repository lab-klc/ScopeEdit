import math
import torch

from .editable_model import EditableModel
from ..utils import _logits
from ...util.lora_layers import LoRALinear, local_seed_scope


class MORE(EditableModel):
    def __init__(self, model, config, model_constructor):
        super().__init__(model, config, model_constructor)
        if not str(self.config.device).startswith("cuda"):
            self.config.device = f"cuda:{self.config.device}"
        self.scopeedit_shared_groups = set()
        self.bridge_shared_groups = self.scopeedit_shared_groups
        self._inject_lora()
        self.group_map = self._collect_groups()
        self._freeze_non_lora()
        self._basis_initialized = False
        self._prepare_basis_init()

    @staticmethod
    def _resolve_attr_path(root, path):
        cur = root
        for part in path.split("."):
            if cur is None or not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    def _wrap_linear_module(self, module, child_name, group, full_name, more_nonlinear, more_layer_norm, more_layer_norm_eps):
        child = getattr(module, child_name, None)
        if child is None or isinstance(child, LoRALinear):
            return False
        if not isinstance(child, torch.nn.Linear):
            return False

        lora_child = LoRALinear(
            child,
            rank=self.config.rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            more_nonlinear=more_nonlinear,
            more_layer_norm=more_layer_norm,
            more_layer_norm_eps=more_layer_norm_eps,
            **self._scopeedit_kwargs(group),
        )
        lora_child.more_group = group
        lora_child.more_name = full_name
        setattr(module, child_name, lora_child)
        return True

    def _wrap_nested_linear(self, root, attr_path, group, prefix, more_nonlinear, more_layer_norm, more_layer_norm_eps):
        parts = attr_path.split(".")
        parent = root
        for part in parts[:-1]:
            if not hasattr(parent, part):
                return False
            parent = getattr(parent, part)
        full_name = f"{prefix}.{attr_path}"
        return self._wrap_linear_module(
            parent,
            parts[-1],
            group,
            full_name,
            more_nonlinear,
            more_layer_norm,
            more_layer_norm_eps,
        )

    def _wrap_vision_block(self, block, idx, prefix, more_nonlinear, more_layer_norm, more_layer_norm_eps):
        group = f"vision_encoder.layer.{idx}"
        wrapped = False
        candidates = [
            "mlp.fc1",
            "mlp.fc2",
            "mlp.c_fc",
            "mlp.c_proj",
        ]
        for attr_path in candidates:
            wrapped = (
                self._wrap_nested_linear(
                    block,
                    attr_path,
                    group,
                    f"{prefix}.{idx}",
                    more_nonlinear,
                    more_layer_norm,
                    more_layer_norm_eps,
                )
                or wrapped
            )
        return wrapped

    def _wrap_qformer_block(self, block, idx, more_nonlinear, more_layer_norm, more_layer_norm_eps):
        group = f"qformer.layer.{idx}"
        wrapped = False
        candidates = [
            "intermediate_query.dense",
            "output_query.dense",
        ]

        for attr_path in candidates:
            wrapped = (
                self._wrap_nested_linear(
                    block,
                    attr_path,
                    group,
                    "Qformer.bert.encoder.layer",
                    more_nonlinear,
                    more_layer_norm,
                    more_layer_norm_eps,
                )
                or wrapped
            )
        return wrapped

    def _inject_optional_vision_lora(self, more_nonlinear, more_layer_norm, more_layer_norm_eps):
        n_vision = (
            max(0, int(getattr(self.config, "more_vision_num_layers", 0) or 0))
            if self._edit_vision_encoder()
            else 0
        )
        n_qformer = max(0, int(getattr(self.config, "more_qformer_num_layers", 0) or 0))
        if n_vision <= 0 and n_qformer <= 0:
            return

        if n_vision > 0:
            vision_paths = []
            if hasattr(self.model, "visual_encoder"):
                vision_paths.extend(
                    [
                        ("visual_encoder.blocks", "visual_encoder.blocks"),
                        ("visual_encoder.transformer.resblocks", "visual_encoder.transformer.resblocks"),
                    ]
                )
            if hasattr(self.model, "model"):
                vision_paths.extend(
                    [
                        ("model.vision_tower.vision_model.encoder.layers", "model.vision_tower.vision_model.encoder.layers"),
                        ("model.vision_tower.encoder.layers", "model.vision_tower.encoder.layers"),
                        ("model.vision_model.encoder.layers", "model.vision_model.encoder.layers"),
                    ]
                )

            layers = None
            prefix = None
            for path, name_prefix in vision_paths:
                resolved = self._resolve_attr_path(self.model, path)
                if resolved is not None:
                    layers = resolved
                    prefix = name_prefix
                    break

            if layers is None:
                print("[M-ORE] Vision layer injection requested, but no vision encoder layers were found.")
            else:
                total_layers = len(layers)
                start_idx = max(0, total_layers - n_vision)
                wrapped_any = False
                for idx in range(start_idx, total_layers):
                    wrapped_any = (
                        self._wrap_vision_block(
                            layers[idx],
                            idx,
                            prefix,
                            more_nonlinear,
                            more_layer_norm,
                            more_layer_norm_eps,
                        )
                        or wrapped_any
                    )
                if not wrapped_any:
                    print("[M-ORE] Vision layer injection matched no safe linear modules.")

        if n_qformer > 0 and hasattr(self.model, "Qformer"):
            layers = self._resolve_attr_path(self.model, "Qformer.bert.encoder.layer")
            if layers is None:
                print("[M-ORE] Q-Former injection requested, but encoder layers were not found.")
            else:
                total_layers = len(layers)
                start_idx = max(0, total_layers - n_qformer)
                wrapped_any = False
                for idx in range(start_idx, total_layers):
                    wrapped_any = (
                        self._wrap_qformer_block(
                            layers[idx],
                            idx,
                            more_nonlinear,
                            more_layer_norm,
                            more_layer_norm_eps,
                        )
                        or wrapped_any
                    )
                if not wrapped_any:
                    print("[M-ORE] Q-Former injection matched no linear modules.")

    @staticmethod
    def _is_vision_side_group(group_name):
        group_name = str(group_name)
        return (
            group_name == "vision_proj"
            or group_name.startswith("vision_encoder.")
            or group_name.startswith("qformer.")
        )

    @staticmethod
    def _is_vision_encoder_group(group_name):
        return str(group_name).startswith("vision_encoder.")

    def _edit_text_llm(self):
        return bool(getattr(self.config, "more_edit_text_llm", True))

    def _edit_projector(self):
        return bool(getattr(self.config, "more_edit_projector", True))

    def _edit_vision_encoder(self):
        return bool(getattr(self.config, "more_edit_vision_encoder", False))

    def _scopeedit_config_value(self, name, default=None):
        scopeedit_name = f"scopeedit_{name}"
        if hasattr(self.config, scopeedit_name):
            value = getattr(self.config, scopeedit_name)
            if value is not None:
                return value
        return getattr(self.config, f"bridge_{name}", default)

    def _scopeedit_enabled(self):
        return bool(self._scopeedit_config_value("enable", False))

    def _scopeedit_rank_shared(self):
        return max(0, int(getattr(self.config, "rank_shared", 0) or 0))

    def _scopeedit_rank_private(self):
        rank_private = getattr(self.config, "rank_private", None)
        if rank_private is None:
            rank_private = int(self.config.rank) - self._scopeedit_rank_shared()
        rank_private = int(rank_private)
        if rank_private + self._scopeedit_rank_shared() != int(self.config.rank):
            raise ValueError(
                f"ScopeEdit expects rank_private + rank_shared == rank, got "
                f"{rank_private} + {self._scopeedit_rank_shared()} != {self.config.rank}"
            )
        return rank_private

    def _scopeedit_kwargs(self, group_name):
        if not self._scopeedit_enabled():
            return {}
        return {
            "scopeedit_enable": True,
            "scopeedit_rank_private": self._scopeedit_rank_private(),
            "scopeedit_rank_shared": self._scopeedit_rank_shared(),
            "scopeedit_modality": "vision" if self._is_vision_side_group(group_name) else "text",
            "scopeedit_pooling": self._scopeedit_config_value("pooling", "mean"),
            "scopeedit_scope_separation": self._scopeedit_config_value("scope_separation", True),
        }

    @staticmethod
    def _group_layer_index(group_name):
        parts = str(group_name).split(".")
        if not parts:
            return None
        try:
            return int(parts[-1])
        except ValueError:
            return None

    def _configure_scopeedit_groups(self):
        if not self._scopeedit_enabled():
            self.scopeedit_shared_groups = set()
            self.bridge_shared_groups = self.scopeedit_shared_groups
            return

        self.scopeedit_shared_groups = set()
        self.bridge_shared_groups = self.scopeedit_shared_groups
        for group_modules in self.group_map.values():
            for module in group_modules:
                if getattr(module, "scopeedit_enabled", getattr(module, "bridge_enabled", False)):
                    module.more_bridge_active = False

    def _select_scopeedit_groups(self, active_groups):
        if not self._scopeedit_enabled():
            return []

        allow_projector = bool(self._scopeedit_config_value("use_projector_layer", False))
        allow_vision_encoder = bool(self._scopeedit_config_value("use_vision_encoder_layer", False))
        eligible_groups = []
        for group in active_groups:
            if self._is_vision_side_group(group):
                if group == "vision_proj" and allow_projector:
                    eligible_groups.append(group)
                elif self._is_vision_encoder_group(group) and allow_vision_encoder:
                    eligible_groups.append(group)
                continue
            eligible_groups.append(group)

        num_layers = int(self._scopeedit_config_value("num_layers", 0) or 0)
        if num_layers <= 0 or num_layers >= len(eligible_groups):
            return list(eligible_groups)
        return list(eligible_groups[:num_layers])

    def _inject_lora(self):
        if getattr(self.model, "_more_lora_injected", False):
            return

        more_nonlinear = getattr(self.config, "more_nonlinear", "none")
        more_layer_norm = getattr(self.config, "more_layer_norm", False)
        more_layer_norm_eps = getattr(self.config, "more_layer_norm_eps", 1e-5)

        if hasattr(self.model, "llama_model") and hasattr(self.model, "llama_proj"):
            lm_model = self.model.llama_model
            proj = self.model.llama_proj
            proj_attr = "llama_proj"
            proj_base = "llama_proj"
            group_prefix = "llama.layer"
            layer_attr = "mlp"
            proj_names = ("up_proj", "down_proj")
            if hasattr(lm_model, "model") and hasattr(lm_model.model, "layers"):
                layers = lm_model.model.layers
                name_prefix = "llama_model.model.layers"
            elif hasattr(lm_model, "layers"):
                layers = lm_model.layers
                name_prefix = "llama_model.layers"
            else:
                raise ValueError("Unsupported LLaMA language model structure for M-ORE.")
        elif hasattr(self.model, "opt_model") and hasattr(self.model, "opt_proj"):
            lm_model = self.model.opt_model
            proj = self.model.opt_proj
            proj_attr = "opt_proj"
            proj_base = "opt_proj"
            group_prefix = "opt.layer"
            layer_attr = None
            proj_names = ("fc1", "fc2")
            if (
                hasattr(lm_model, "model")
                and hasattr(lm_model.model, "decoder")
                and hasattr(lm_model.model.decoder, "layers")
            ):
                layers = lm_model.model.decoder.layers
                name_prefix = "opt_model.model.decoder.layers"
            elif hasattr(lm_model, "decoder") and hasattr(lm_model.decoder, "layers"):
                layers = lm_model.decoder.layers
                name_prefix = "opt_model.decoder.layers"
            else:
                raise ValueError("Unsupported OPT language model structure for M-ORE.")
        else:
            raise ValueError(
                "M-ORE trainer supports models with llama_model/llama_proj (MiniGPT4/LLaVA) "
                "or opt_model/opt_proj (BLIP-2 OPT)."
            )

        n_layers = len(layers)
        n_last = max(0, int(self.config.n_last_layers))
        start_idx = max(0, n_layers - n_last)

        if self._edit_text_llm():
            for idx in range(start_idx, n_layers):
                layer = layers[idx]
                for proj_name in proj_names:
                    linear = getattr(layer.mlp, proj_name) if layer_attr else getattr(layer, proj_name)
                    if isinstance(linear, LoRALinear):
                        continue
                    lora = LoRALinear(
                        linear,
                        rank=self.config.rank,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        more_nonlinear=more_nonlinear,
                        more_layer_norm=more_layer_norm,
                        more_layer_norm_eps=more_layer_norm_eps,
                        **self._scopeedit_kwargs(f"{group_prefix}.{idx}"),
                    )
                    lora.more_group = f"{group_prefix}.{idx}"
                    if layer_attr:
                        lora.more_name = f"{name_prefix}.{idx}.mlp.{proj_name}"
                        setattr(layer.mlp, proj_name, lora)
                    else:
                        lora.more_name = f"{name_prefix}.{idx}.{proj_name}"
                        setattr(layer, proj_name, lora)

        if self._edit_projector():
            if isinstance(proj, LoRALinear):
                pass
            elif isinstance(proj, torch.nn.Linear):
                lora_proj = LoRALinear(
                    proj,
                    rank=self.config.rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    more_nonlinear=more_nonlinear,
                    more_layer_norm=more_layer_norm,
                    more_layer_norm_eps=more_layer_norm_eps,
                    **self._scopeedit_kwargs("vision_proj"),
                )
                lora_proj.more_group = "vision_proj"
                lora_proj.more_name = proj_base
                setattr(self.model, proj_attr, lora_proj)
            else:
                replaced = False

                def _wrap_proj(module, prefix=""):
                    nonlocal replaced
                    for child_name, child in module.named_children():
                        full_name = f"{prefix}.{child_name}" if prefix else child_name
                        if isinstance(child, LoRALinear):
                            continue
                        if isinstance(child, torch.nn.Linear):
                            lora_child = LoRALinear(
                                child,
                                rank=self.config.rank,
                                lora_alpha=self.config.lora_alpha,
                                lora_dropout=self.config.lora_dropout,
                                more_nonlinear=more_nonlinear,
                                more_layer_norm=more_layer_norm,
                                more_layer_norm_eps=more_layer_norm_eps,
                                **self._scopeedit_kwargs("vision_proj"),
                            )
                            lora_child.more_group = "vision_proj"
                            lora_child.more_name = f"{proj_base}.{full_name}"
                            setattr(module, child_name, lora_child)
                            replaced = True
                        else:
                            _wrap_proj(child, full_name)

                _wrap_proj(proj)
                if not replaced:
                    raise ValueError(f"Unsupported {proj_base} type for M-ORE LoRA injection.")

        self._inject_optional_vision_lora(
            more_nonlinear=more_nonlinear,
            more_layer_norm=more_layer_norm,
            more_layer_norm_eps=more_layer_norm_eps,
        )
        self.model._more_lora_injected = True

    def _collect_groups(self):
        group_map = {}
        for module in self.model.modules():
            if isinstance(module, LoRALinear) and module.more_group is not None:
                group_map.setdefault(module.more_group, []).append(module)
        return group_map

    def _freeze_non_lora(self):
        freeze_a = getattr(self.config, "more_freeze_A", True)
        for name, p in self.model.named_parameters():
            if "lora_A" in name:
                p.requires_grad = not freeze_a
            elif "lora_B" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _iter_lora_modules(self):
        for modules in self.group_map.values():
            for module in modules:
                yield module

    def _basis_init_name(self):
        basis_init = str(getattr(self.config, "more_basis_init", "orthogonal")).lower()
        return "orthogonal" if basis_init in ("", "default") else basis_init

    def _prepare_basis_init(self):
        if self._scopeedit_enabled():
            self._basis_initialized = False
            return
        basis_init = self._basis_init_name()
        if basis_init in ("", "orthogonal", "default"):
            with local_seed_scope(getattr(self.config, "seed", None)):
                for module in self._iter_lora_modules():
                    module.reset_lora_A("orthogonal")
            print(
                f"[M-ORE] Using orthogonal LoRA basis init "
                f"(seed={getattr(self.config, 'seed', None)})."
            )
            self._basis_initialized = True
            return
        if basis_init in ("gaussian", "xavier"):
            with local_seed_scope(getattr(self.config, "seed", None)):
                for module in self._iter_lora_modules():
                    module.reset_lora_A(basis_init)
            print(
                f"[M-ORE] Using {basis_init} LoRA basis init "
                f"(seed={getattr(self.config, 'seed', None)})."
            )
            self._basis_initialized = True
            return
        if basis_init in ("pca", "svd"):
            self._basis_initialized = False
            return
        raise ValueError(f"Unsupported M-ORE basis init: {basis_init}")

    def initialize_basis(self, loader):
        if self._scopeedit_enabled():
            if self._basis_initialized:
                return
            basis_init = self._basis_init_name()
            if basis_init not in ("orthogonal", "gaussian", "xavier", "pca", "svd"):
                raise ValueError(
                    f"Unsupported ScopeEdit basis init: {basis_init}. "
                    "Use orthogonal, gaussian, xavier, pca, or svd."
                )
            modules = [
                m
                for m in self._iter_lora_modules()
                if getattr(m, "scopeedit_enabled", getattr(m, "bridge_enabled", False))
            ]

            if basis_init in ("pca", "svd"):
                basis_batch_key = str(getattr(self.config, "more_basis_batch_key", "edit_inner"))
                basis_batches = max(1, int(getattr(self.config, "more_basis_num_batches", 8)))
                max_samples = max(1, int(getattr(self.config, "more_basis_max_samples", 512)))

                print(
                    f"[ScopeEdit] Collecting activations for {basis_init} private/shared bases "
                    f"(batch_key={basis_batch_key}, batches={basis_batches}, max_samples={max_samples})..."
                )

                for module in modules:
                    module.reset_basis_cache(max_samples)
                    module.more_collect_bridge_basis = True
                    module._zero_lora_A()
                    module._zero_lora_B()
                    module.more_bridge_strength.fill_(1.0)

                old_mode = self.model.training
                self.model.eval()
                self.model.zero_grad(set_to_none=True)

                processed = 0
                for batch in loader:
                    if processed >= basis_batches:
                        break

                    if isinstance(batch, dict):
                        if basis_batch_key not in batch:
                            raise KeyError(
                                f"Batch key '{basis_batch_key}' not found while building ScopeEdit basis."
                            )
                        basis_batch = batch[basis_batch_key]
                    else:
                        basis_batch = batch

                    target_lens = self._compute_target_lens(basis_batch)
                    LoRALinear.more_use_masked_z = bool(getattr(self.config, "more_use_masked_z", True))
                    LoRALinear.more_target_lens = target_lens
                    LoRALinear.more_batch_size = (
                        len(target_lens) if isinstance(target_lens, (list, tuple)) else None
                    )
                    with torch.no_grad():
                        self.model(basis_batch)
                    LoRALinear.more_target_lens = None
                    LoRALinear.more_batch_size = None
                    LoRALinear.more_use_masked_z = True

                    processed += 1
                    if all(module.basis_cache_size() >= max_samples for module in modules):
                        break

                initialized = 0
                with local_seed_scope(getattr(self.config, "seed", None)):
                    for module in modules:
                        module.more_collect_bridge_basis = False
                        if module.initialize_scopeedit_basis_from_data(basis_init):
                            initialized += 1
                        module._zero_lora_B()
                        module.clear_basis_cache()

                self.model.zero_grad(set_to_none=True)
                self.model.train(old_mode)
                self._configure_scopeedit_groups()
                self._basis_initialized = True

                print(
                    f"[ScopeEdit] Initialized {initialized}/{len(modules)} modules with "
                    f"{basis_init} private/shared bases; seed={getattr(self.config, 'seed', None)}."
                )
                return

            with local_seed_scope(getattr(self.config, "seed", None)):
                for module in modules:
                    module.reset_lora_A(basis_init)
                    module._zero_lora_B()
                    module.more_bridge_strength.fill_(1.0)

            self._configure_scopeedit_groups()
            self._basis_initialized = True

            print(
                f"[ScopeEdit] Using frozen {basis_init} shared/private bases; "
                f"seed={getattr(self.config, 'seed', None)}; "
                "shared groups will be selected from more/top_groups each step."
            )
            return

        basis_init = self._basis_init_name()
        if self._basis_initialized or basis_init not in ("pca", "svd"):
            return

        basis_batch_key = str(getattr(self.config, "more_basis_batch_key", "edit_inner"))
        basis_batches = max(1, int(getattr(self.config, "more_basis_num_batches", 8)))
        max_samples = max(1, int(getattr(self.config, "more_basis_max_samples", 512)))

        print(
            f"[M-ORE] Collecting activations for {basis_init} basis "
            f"(batch_key={basis_batch_key}, batches={basis_batches}, max_samples={max_samples})..."
        )

        modules = list(self._iter_lora_modules())
        for module in modules:
            module.reset_basis_cache(max_samples)
            module.more_collect_basis = True

        old_mode = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        processed = 0
        for batch in loader:
            if processed >= basis_batches:
                break

            if isinstance(batch, dict):
                if basis_batch_key not in batch:
                    raise KeyError(
                        f"Batch key '{basis_batch_key}' not found while building M-ORE basis."
                    )
                basis_batch = batch[basis_batch_key]
            else:
                basis_batch = batch

            target_lens = self._compute_target_lens(basis_batch)
            LoRALinear.more_use_masked_z = bool(getattr(self.config, "more_use_masked_z", True))
            LoRALinear.more_target_lens = target_lens
            LoRALinear.more_batch_size = (
                len(target_lens) if isinstance(target_lens, (list, tuple)) else None
            )
            with torch.no_grad():
                self.model(basis_batch)
            LoRALinear.more_target_lens = None
            LoRALinear.more_batch_size = None
            LoRALinear.more_use_masked_z = True

            processed += 1
            if all(module.basis_cache_size() >= max_samples for module in modules):
                break

        initialized = 0
        for module in modules:
            module.more_collect_basis = False
            if module.initialize_lora_A_from_data(basis_init):
                initialized += 1
            module.clear_basis_cache()

        self.model.zero_grad(set_to_none=True)
        self.model.train(old_mode)
        self._basis_initialized = True

        print(
            f"[M-ORE] Initialized {initialized}/{len(modules)} LoRA modules with {basis_init} basis."
        )

    def outer_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _compute_target_lens(self, batch):
        if hasattr(self.model, "compute_more_target_lens"):
            lens = self.model.compute_more_target_lens(batch)
            if lens is not None:
                return lens
        text_inputs = batch.get("text_input", None)
        prompts_len = batch.get("prompts_len", None)
        if (
            isinstance(text_inputs, list)
            and isinstance(prompts_len, list)
            and len(text_inputs) == len(prompts_len)
            and hasattr(self.model, "llama_tokenizer")
        ):
            lens = []
            for text, plen in zip(text_inputs, prompts_len):
                tlen = len(self.model.llama_tokenizer.encode(text, add_special_tokens=False))
                lens.append(max(0, tlen - int(plen)))
            return lens

        labels = batch.get("labels", None)
        if isinstance(labels, torch.Tensor):
            tlen = labels.size(1)
            return [tlen for _ in range(labels.size(0))]
        return None

    def _compute_rls_delta(self, module, eta, rls_lambda):
        gA = module.lora_A.weight.grad
        gB = module.lora_B.weight.grad
        if gA is None and gB is None:
            return None, None

        z = module.more_last_z
        if z is None:
            return None, None

        P = module.more_P
        z = z.to(dtype=torch.float32)
        P = P.to(dtype=torch.float32)

        denom = rls_lambda + torch.dot(z, P @ z)
        if denom.abs().item() < 1e-12:
            return None, None

        Pz = P @ z
        P = P - torch.outer(Pz, Pz) / denom
        module.more_P.copy_(P)

        delta_A = (P @ gA.float()) * eta if gA is not None else None
        delta_B = (gB.float() @ P) * eta if gB is not None else None
        return delta_A, delta_B

    def _locality_key_enabled(self):
        return bool(getattr(self.config, "more_accumulate_locality_keys", False))

    def _locality_key_groups(self, active_groups):
        group_mode = str(getattr(self.config, "more_locality_key_groups", "active")).lower()
        if group_mode == "all":
            return list(self.group_map.keys())
        if group_mode != "active":
            raise ValueError(
                f"Unsupported more_locality_key_groups={group_mode}; use active or all."
            )
        if active_groups is None:
            return list(self.group_map.keys())
        return [g for g in active_groups if g in self.group_map]

    def _private_lambda_for_module(self, module):
        locality_lambda = getattr(self.config, "more_locality_key_lambda", None)
        if locality_lambda is not None:
            return locality_lambda

        scopeedit_lambda_private = self._scopeedit_config_value("lambda_private", None)
        if scopeedit_lambda_private is not None:
            return scopeedit_lambda_private

        rls_vision = getattr(self.config, "more_rls_lambda_vision", None)
        if self._is_vision_side_group(module.more_group) and rls_vision is not None:
            return rls_vision
        return self.config.rls_lambda

    def _shared_lambda_for_module(self, module, lambda_private):
        locality_lambda_shared = getattr(self.config, "more_locality_key_lambda_shared", None)
        if locality_lambda_shared is not None:
            return locality_lambda_shared

        scopeedit_lambda_shared = self._scopeedit_config_value("lambda_shared", None)
        if scopeedit_lambda_shared is not None:
            return scopeedit_lambda_shared
        return lambda_private

    def accumulate_current_keys(self, active_groups=None, source=None):
        if not self._locality_key_enabled():
            return {}

        groups = self._locality_key_groups(active_groups)
        include_shared = bool(getattr(self.config, "more_locality_key_include_shared", False))
        updated_private = 0
        updated_shared = 0
        for group in groups:
            for module in self.group_map.get(group, []):
                lambda_private = self._private_lambda_for_module(module)
                lambda_shared = self._shared_lambda_for_module(module, lambda_private)
                result = module.accumulate_current_key(
                    rls_lambda_private=lambda_private,
                    rls_lambda_shared=lambda_shared,
                    include_shared=include_shared,
                )
                updated_private += int(bool(result.get("updated_private", False)))
                updated_shared += int(bool(result.get("updated_shared", False)))

        prefix = "more/locality_key"
        stats = {
            f"{prefix}_private_updates": float(updated_private),
            f"{prefix}_shared_updates": float(updated_shared),
        }
        if source:
            stats[f"{prefix}_{source}_private_updates"] = float(updated_private)
            stats[f"{prefix}_{source}_shared_updates"] = float(updated_shared)
        return stats

    def forward(self, *inputs, **kwargs):
        if ("minigpt4" in self.config.model_name.lower()
                or "blip" in self.config.model_name.lower()
                or "llava" in self.config.model_name.lower()):
            outputs = self.model(*inputs, **kwargs)
        elif "gpt" in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"]))
        elif "llama" in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"]))
        elif "qwen" in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"]))
        else:
            outputs = _logits(self.model(**kwargs))
        return outputs

    def edit(self, batch, condition=None, detach_history=False, return_factors=False, **kwargs):
        use_masked = getattr(self.config, "more_use_masked_z", True)
        LoRALinear.more_use_masked_z = bool(use_masked)
        target_lens = self._compute_target_lens(batch)
        LoRALinear.more_target_lens = target_lens
        LoRALinear.more_batch_size = (
            len(target_lens) if isinstance(target_lens, (list, tuple)) else None
        )

        outputs = self.model(batch)
        LoRALinear.more_target_lens = None
        LoRALinear.more_batch_size = None
        LoRALinear.more_use_masked_z = True

        if not isinstance(outputs, torch.Tensor):
            logits = outputs.logits
            labels = outputs.labels
        else:
            logits = outputs
            labels = batch["labels"]

        loss = self.edit_loss_fn(self.config, logits, labels, multimodal=True)["nll"]
        loss.backward()

        score_norm = getattr(self.config, "more_score_norm", "none")
        score_norm = score_norm.lower() if isinstance(score_norm, str) else "none"
        scores = {}
        for group, modules in self.group_map.items():
            scores[group] = sum(module.more_score_value(score_norm) for module in modules)

        ranked_groups = [g for g, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        k = min(self.config.top_k, len(ranked_groups))
        active_groups = ranked_groups[:k] if k > 0 else []

        if getattr(self.config, "more_force_vision_encoder", False):
            forced_groups = [g for g in ranked_groups if self._is_vision_encoder_group(g)]
            active_groups = forced_groups + [g for g in active_groups if g not in forced_groups]
            if k > 0:
                for g in ranked_groups:
                    if g not in active_groups:
                        active_groups.append(g)
                    if len(active_groups) >= max(k, len(forced_groups)):
                        break
                active_groups = active_groups[: max(k, len(forced_groups))]

        update_mode = getattr(self.config, "more_update_mode", "online").lower()
        restore_data = []
        restore_p = getattr(self.config, "more_restore_p", True)
        if update_mode == "temporary":
            for group in active_groups:
                for module in self.group_map.get(group, []):
                    restore_data.append((module, module.snapshot_more_state(restore_p)))

        scopeedit_shared_groups = self._select_scopeedit_groups(active_groups)
        self.scopeedit_shared_groups = set(scopeedit_shared_groups)
        self.bridge_shared_groups = self.scopeedit_shared_groups
        for group, modules in self.group_map.items():
            shared_active = group in self.scopeedit_shared_groups
            for module in modules:
                if getattr(module, "scopeedit_enabled", getattr(module, "bridge_enabled", False)):
                    module.more_bridge_active = shared_active

        eta_vision = getattr(self.config, "more_eta_vision", None)
        rls_vision = getattr(self.config, "more_rls_lambda_vision", None)
        scopeedit_eta_shared = self._scopeedit_config_value("eta_shared", None)
        scopeedit_lambda_private = self._scopeedit_config_value("lambda_private", None)
        scopeedit_lambda_shared = self._scopeedit_config_value("lambda_shared", None)
        scopeedit_tau = float(self._scopeedit_config_value("tau", 0.0))
        scopeedit_beta = float(self._scopeedit_config_value("beta", 10.0))
        scopeedit_scope_mode = str(self._scopeedit_config_value("scope_mode", "gated"))
        scopeedit_stats = []
        for group in active_groups:
            for module in self.group_map.get(group, []):
                use_vision_hparams = self._is_vision_side_group(module.more_group)
                eta_private = (
                    eta_vision if (use_vision_hparams and eta_vision is not None) else self.config.eta
                )
                lambda_private = (
                    scopeedit_lambda_private
                    if scopeedit_lambda_private is not None
                    else (
                        rls_vision
                        if (use_vision_hparams and rls_vision is not None)
                        else self.config.rls_lambda
                    )
                )
                lambda_shared = (
                    scopeedit_lambda_shared if scopeedit_lambda_shared is not None else lambda_private
                )
                eta_shared = scopeedit_eta_shared if scopeedit_eta_shared is not None else eta_private
                result = module.apply_online_update(
                    eta_private=eta_private,
                    lambda_private=lambda_private,
                    eta_shared=eta_shared,
                    lambda_shared=lambda_shared,
                    scopeedit_tau=scopeedit_tau,
                    scopeedit_beta=scopeedit_beta,
                    scopeedit_scope_mode=scopeedit_scope_mode,
                    enable_shared=group in self.scopeedit_shared_groups,
                )
                scopeedit_stats.append(result)

        self.model.zero_grad(set_to_none=True)

        info_dict = {"more/top_groups": active_groups}
        if self._scopeedit_enabled():
            info_dict["scopeedit/shared_groups"] = scopeedit_shared_groups
            if scopeedit_stats:
                info_dict["scopeedit/avg_alpha"] = sum(stat["alpha"] for stat in scopeedit_stats) / len(scopeedit_stats)
                info_dict["scopeedit/avg_gamma"] = sum(stat["gamma"] for stat in scopeedit_stats) / len(scopeedit_stats)
                info_dict["scopeedit/avg_cos"] = sum(stat["cos"] for stat in scopeedit_stats) / len(scopeedit_stats)
                info_dict["scopeedit/avg_support"] = sum(stat["support"] for stat in scopeedit_stats) / len(scopeedit_stats)
                info_dict["scopeedit/shared_updates"] = float(
                    sum(1 for stat in scopeedit_stats if stat["updated_shared"])
                )
                info_dict["scopeedit/private_updates"] = float(
                    sum(1 for stat in scopeedit_stats if stat["updated_private"])
                )
        if update_mode == "temporary" and restore_data:
            def _restore():
                for module, state in restore_data:
                    module.restore_more_state(state)

            info_dict["restore_fn"] = _restore

        return self, info_dict

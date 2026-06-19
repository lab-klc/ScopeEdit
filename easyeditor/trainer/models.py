import logging

import torch
import torch.nn as nn

from .llava_models.llava_v15 import LlavaV15ForEditing

LOG = logging.getLogger(__name__)


def _scopeedit_config_value(config, name, default=None):
    scopeedit_name = f"scopeedit_{name}"
    if hasattr(config, scopeedit_name):
        value = getattr(config, scopeedit_name)
        if value is not None:
            return value
    return getattr(config, f"bridge_{name}", default)


def _resolve_inner_params_by_suffix(inner_params, param_names):
    resolved = []
    unresolved = []
    for name in inner_params:
        if name in param_names:
            resolved.append(name)
            continue

        parts = name.split(".")
        if "layers" in parts:
            suffix = ".".join(parts[parts.index("layers"):])
        else:
            suffix = ".".join(parts[-5:])

        matches = [p for p in param_names if p.endswith(suffix)]
        if len(matches) == 1:
            resolved.append(matches[0])
        else:
            unresolved.append((name, matches))

    return resolved, unresolved


def _build_blip2_model(config):
    from .blip2_models.blip2_opt import Blip2OPT

    model_kwargs = {
        "vit_model": "eva_clip_g",
        "img_size": 364,
        "use_grad_checkpoint": True,
        "vit_precision": "fp32",
        "freeze_vit": getattr(config, "freeze_vit", True),
        "freeze_qformer": config.freeze_qformer,
        "opt_model": config.name,
        "state_dict_file": config.state_dict_file,
        "qformer_name_or_path": config.qformer_name_or_path,
        "qformer_checkpoint": config.qformer_checkpoint,
    }
    opt_precision = getattr(config, "opt_precision", None)
    if opt_precision:
        model_kwargs["opt_precision"] = opt_precision
    try:
        return Blip2OPT(**model_kwargs)
    except TypeError:
        model_kwargs.pop("opt_precision", None)
        return Blip2OPT(**model_kwargs)


def _build_minigpt4_model(config):
    from .blip2_models.mini_gpt4 import MiniGPT4

    return MiniGPT4(
        vit_model="eva_clip_g",
        qformer_checkpoint=config.qformer_checkpoint,
        img_size=364,
        use_grad_checkpoint=True,
        vit_precision="fp32",
        freeze_vit=getattr(config, "freeze_vit", True),
        freeze_qformer=config.freeze_qformer,
        llama_model=config.name,
        state_dict_file=config.state_dict_file,
        qformer_name_or_path=config.qformer_name_or_path,
        pretrained_ckpt=config.pretrained_ckpt,
    )


def _build_llava_model(config):
    model_path = (
        config.name
        if config.model_name.lower() in ("llava1.5", "llava-v1.5") and hasattr(config, "name")
        else config.model_name
    )
    torch_dtype = torch.float16 if getattr(config, "half", False) else torch.float32
    device_map = "auto" if getattr(config, "model_parallel", False) else None
    llava_kwargs = {
        "model_name_or_path": model_path,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "use_chat_template": getattr(config, "use_chat_template", True),
        "scopeedit_visual_anchor_enable": _scopeedit_config_value(config, "visual_anchor_enable", False),
        "scopeedit_visual_anchor_start": _scopeedit_config_value(config, "visual_anchor_start", None),
        "scopeedit_visual_anchor_end": _scopeedit_config_value(config, "visual_anchor_end", None),
    }
    image_proc = getattr(config, "image_processor_name_or_path", None)
    if image_proc is not None:
        llava_kwargs["image_processor_name_or_path"] = image_proc
    try:
        return LlavaV15ForEditing(**llava_kwargs)
    except TypeError:
        llava_kwargs.pop("image_processor_name_or_path", None)
        return LlavaV15ForEditing(**llava_kwargs)


def get_model(config):
    model_name = config.model_name.lower()
    if config.model_name == "blip2":
        model = _build_blip2_model(config)
    elif config.model_name == "minigpt4":
        model = _build_minigpt4_model(config)
    elif "llava" in model_name and "onevision" not in model_name:
        model = _build_llava_model(config)
    else:
        raise NotImplementedError(
            "ScopeEdit/M-ORE entry supports blip2, minigpt4, and LLaVA-v1.5 models."
        )

    if config.dropout is not None:
        n_reset = 0
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = config.dropout
                n_reset += 1
            if hasattr(module, "dropout") and isinstance(module.dropout, float):
                module.dropout = config.dropout
                n_reset += 1
            if hasattr(module, "activation_dropout") and isinstance(module.activation_dropout, float):
                module.activation_dropout = config.dropout
                n_reset += 1
        LOG.info("Set %s dropout modules to p=%s", n_reset, config.dropout)

    param_names = [name for name, _ in model.named_parameters()]
    bad_inner_params = [p for p in config.inner_params if p not in param_names]
    if bad_inner_params and "llava" in model_name:
        resolved, unresolved = _resolve_inner_params_by_suffix(config.inner_params, param_names)
        if not unresolved:
            config.inner_params = resolved
            LOG.info("Remapped inner_params for LLaVA using suffix matches: %s", config.inner_params)
            bad_inner_params = []
        else:
            bad_inner_params = [p for p, _ in unresolved]
    if bad_inner_params:
        raise ValueError(f"Params {bad_inner_params} do not exist in model of type {type(model)}.")

    return model

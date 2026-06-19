from dataclasses import dataclass, fields
from typing import Optional, Any, List
import yaml

from ...util.hparams import HyperParams


_SCOPEEDIT_LEGACY_KEY_MAP = {
    "bridge_enable": "scopeedit_enable",
    "bridge_num_layers": "scopeedit_num_layers",
    "bridge_tau": "scopeedit_tau",
    "bridge_beta": "scopeedit_beta",
    "bridge_scope_mode": "scopeedit_scope_mode",
    "bridge_lambda_private": "scopeedit_lambda_private",
    "bridge_lambda_shared": "scopeedit_lambda_shared",
    "bridge_eta_shared": "scopeedit_eta_shared",
    "bridge_pooling": "scopeedit_pooling",
    "bridge_scope_separation": "scopeedit_scope_separation",
    "bridge_use_projector_layer": "scopeedit_use_projector_layer",
    "bridge_use_vision_encoder_layer": "scopeedit_use_vision_encoder_layer",
    "bridge_visual_anchor_enable": "scopeedit_visual_anchor_enable",
    "bridge_visual_anchor_start": "scopeedit_visual_anchor_start",
    "bridge_visual_anchor_end": "scopeedit_visual_anchor_end",
}


@dataclass
class MOREMultimodalTrainingHparams(HyperParams):
    # M-ORE
    rank: int
    lora_alpha: float
    lora_dropout: float
    top_k: int
    eta: float
    rls_lambda: float
    n_last_layers: int
    more_update_mode: str
    more_use_masked_z: bool
    more_restore_p: bool
    more_score_norm: str
    more_eta_vision: Optional[float]
    more_rls_lambda_vision: Optional[float]

    # Multimodal
    qformer_name_or_path: str
    state_dict_file: str

    # Image_dir
    coco_image: str
    rephrase_image: str

    # Model
    name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: List[str]

    archive: Any

    # Method
    alg: str
    lr: float
    seed: int
    debug: bool
    cedit: float
    iedit: float
    cloc: float
    cbase: float
    dropout: float
    train_base: bool
    no_grad_layers: Any

    # Output
    results_dir: str

    # Train
    device: str
    batch_size: int
    model_save_pt: int
    silent: bool
    log_interval: int
    eval_log_interval: int
    final_eval: bool
    val_interval: int
    early_stop_patience: int
    early_stop_key: str
    eval_only: bool
    half: bool
    save: bool
    verbose: bool

    val_batch_size: int
    accumulate_bs: int
    val_steps: int
    opt: str
    grad_clip: float

    qformer_checkpoint: Optional[str] = None
    exact_match: bool = False
    model_parallel: bool = False
    freeze_qformer: bool = True
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None
    pretrained_ckpt: Optional[str] = None
    opt_precision: Optional[str] = None
    use_chat_template: bool = False
    image_processor_name_or_path: Optional[str] = None
    more_basis_init: str = "orthogonal"
    more_basis_num_batches: int = 8
    more_basis_max_samples: int = 512
    more_basis_batch_key: str = "edit_inner"
    more_nonlinear: str = "none"
    more_layer_norm: bool = False
    more_layer_norm_eps: float = 1e-5
    more_edit_text_llm: bool = True
    more_edit_projector: bool = True
    more_edit_vision_encoder: bool = False
    more_vision_num_layers: int = 0
    more_qformer_num_layers: int = 0
    more_force_vision_encoder: bool = False
    freeze_vit: bool = True
    scopeedit_enable: bool = False
    rank_private: Optional[int] = None
    rank_shared: int = 0
    scopeedit_num_layers: int = 0
    scopeedit_tau: float = 0.0
    scopeedit_beta: float = 10.0
    scopeedit_scope_mode: str = "gated"
    scopeedit_lambda_private: Optional[float] = None
    scopeedit_lambda_shared: Optional[float] = None
    scopeedit_eta_shared: Optional[float] = None
    scopeedit_pooling: str = "mean"
    scopeedit_scope_separation: bool = True
    scopeedit_use_projector_layer: bool = True
    scopeedit_use_vision_encoder_layer: bool = False
    scopeedit_visual_anchor_enable: bool = False
    scopeedit_visual_anchor_start: Optional[int] = None
    scopeedit_visual_anchor_end: Optional[int] = None
    more_accumulate_locality_keys: bool = False
    more_locality_key_lambda: Optional[float] = None
    more_locality_key_lambda_shared: Optional[float] = None
    more_locality_key_groups: str = "active"
    more_locality_key_include_shared: bool = False
    more_accumulate_locality_keys_in_training: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        if not config or config.get("alg") != "MORE":
            raise ValueError(
                f"MOREMultimodalTrainingHparams can not load from {hparams_name_or_path}, "
                f"alg_name is {config.get('alg') if config else None}"
            )

        for legacy_key, scopeedit_key in _SCOPEEDIT_LEGACY_KEY_MAP.items():
            if legacy_key in config and scopeedit_key not in config:
                config[scopeedit_key] = config[legacy_key]
            config.pop(legacy_key, None)

        allowed_keys = {field.name for field in fields(cls)}
        unused_keys = sorted(set(config) - allowed_keys)
        if unused_keys:
            print(f">>> [Hparams] Ignoring unused MORE keys: {unused_keys}")
        config = {key: value for key, value in config.items() if key in allowed_keys}
        return cls(**config)

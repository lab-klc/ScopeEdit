from typing import Dict, Optional

import torch

from easyeditor.trainer.utils import RunningStatAverager


def _scopeedit_attr(hparams, name, default=None):
    scopeedit_name = f"scopeedit_{name}"
    if hasattr(hparams, scopeedit_name):
        value = getattr(hparams, scopeedit_name)
        if value is not None:
            return value
    # Backward compatibility for older config objects.
    return getattr(hparams, f"bridge_{name}", default)


class _SliceDataset:
    def __init__(self, base, size):
        self.base = base
        self.size = min(size, len(base))
        self.collate_fn = getattr(base, "collate_fn", None)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.base[idx]


def _format_metric(stats: Dict[str, Optional[float]], key: str):
    val = stats.get(key)
    if val is None:
        return "n/a"
    if isinstance(val, torch.Tensor):
        val = val.item()
    return f"{float(val):.4f}"


def _print_run_config(label: str, config_path: str, hparams, warmup_iters: int, n_edits: int, edit_steps):
    print(f">>> [{label}] Loading configuration from: {config_path}")
    print(
        f">>> [{label}] Effective MORE config | "
        f"scopeedit_enable={_scopeedit_attr(hparams, 'enable', None)} "
        f"basis={getattr(hparams, 'more_basis_init', None)} "
        f"seed={getattr(hparams, 'seed', None)} "
        f"rank={getattr(hparams, 'rank', None)} "
        f"top_k={getattr(hparams, 'top_k', None)} "
        f"eta={getattr(hparams, 'eta', None)} "
        f"rls_lambda={getattr(hparams, 'rls_lambda', None)} "
        f"n_last_layers={getattr(hparams, 'n_last_layers', None)} "
        f"layer_norm={getattr(hparams, 'more_layer_norm', None)} "
        f"edit_text={getattr(hparams, 'more_edit_text_llm', None)} "
        f"edit_projector={getattr(hparams, 'more_edit_projector', None)} "
        f"edit_vision={getattr(hparams, 'more_edit_vision_encoder', None)} "
        f"vision_layers={getattr(hparams, 'more_vision_num_layers', None)}"
    )
    print(
        f">>> [{label}] Runtime | "
        f"warmup_iters={warmup_iters} "
        f"n_edits={n_edits} "
        f"edit_steps={edit_steps} "
        f"update_mode={getattr(hparams, 'more_update_mode', None)} "
        f"restore_p={getattr(hparams, 'more_restore_p', None)} "
        f"archive={getattr(hparams, 'archive', None)} "
        f"results_dir={getattr(hparams, 'results_dir', None)}"
    )


def _print_lora_summary(label: str, trainer):
    iter_lora = getattr(trainer.model, "_iter_lora_modules", None)
    if not callable(iter_lora):
        return
    modules = list(iter_lora())
    scopeedit_modules = sum(
        1
        for module in modules
        # Older LoRA wrappers expose bridge_enabled; new wrappers expose scopeedit_enabled.
        if getattr(module, "scopeedit_enabled", getattr(module, "bridge_enabled", False))
    )
    groups = getattr(trainer.model, "group_map", {})
    print(
        f">>> [{label}] Injected LoRA modules | "
        f"total={len(modules)} "
        f"scopeedit_enabled={scopeedit_modules} "
        f"groups={len(groups)}"
    )


def _print_lifelong(step: int, stats: Dict[str, Optional[float]]):
    rel = _format_metric(stats, "inner/acc_val")
    t_gen = _format_metric(stats, "edit/acc_val")
    m_gen = _format_metric(stats, "image_rephrase/acc_val")
    t_loc = _format_metric(stats, "loc/acc_val")
    m_loc = _format_metric(stats, "image_loc/acc_val")
    t_loc_ans = _format_metric(stats, "loc/acc_answer_val")
    m_loc_ans = _format_metric(stats, "image_loc/acc_answer_val")
    print(
        ">>> [Lifelong] "
        f"{step} edits | "
        f"inner/acc={rel} "
        f"edit/acc={t_gen} "
        f"image_rephrase/acc={m_gen} "
        f"loc/acc={t_loc} "
        f"image_loc/acc={m_loc} "
        f"loc/acc_answer={t_loc_ans} "
        f"image_loc/acc_answer={m_loc_ans}"
    )


def _freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False


def _extract_logits_labels(outputs, batch):
    if isinstance(outputs, torch.Tensor):
        return outputs, batch["labels"]
    return outputs.logits, outputs.labels


def _topk_indices(logits, k):
    return torch.topk(torch.nn.functional.softmax(logits, dim=-1), k=k, dim=-1).indices


def _answer_mask_from_outputs(outputs, logits):
    if isinstance(outputs, torch.Tensor):
        return None
    labels = getattr(outputs, "labels", None)
    if isinstance(labels, torch.Tensor) and isinstance(logits, torch.Tensor):
        if labels.shape[:2] == logits.shape[:2]:
            return labels != -100
    return None


def _locality_pack(outputs, batch, k):
    logits, _ = _extract_logits_labels(outputs, batch)
    topk = _topk_indices(logits, k).detach().cpu()
    pred = logits.argmax(dim=-1).detach().cpu()
    mask = _answer_mask_from_outputs(outputs, logits)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()
    return topk, pred, mask


def _collect_locality_baseline(trainer, batch):
    model = trainer.model
    model.eval()
    with torch.no_grad():
        loc_out = model(batch["loc"])
        loc_image_out = model(batch["loc_image"])

    loc_topk, loc_pred, loc_mask = _locality_pack(loc_out, batch["loc"], 1)
    loc_image_topk, loc_image_pred, loc_image_mask = _locality_pack(
        loc_image_out, batch["loc_image"], 10
    )
    return loc_topk, loc_pred, loc_mask, loc_image_topk, loc_image_pred, loc_image_mask


def _consistency_acc(post_topk, base_topk):
    matches = post_topk.view(-1) == base_topk.view(-1)
    return matches.float().mean()


def _masked_consistency_from_preds(post_pred, base_pred, mask):
    if mask is None:
        return None
    denom = mask.float().sum()
    if denom.item() <= 0:
        return None
    return ((post_pred == base_pred) & mask).float().sum() / denom


def _eval_batch_no_edit(
    trainer,
    batch,
    base_loc_topk=None,
    base_loc_image_topk=None,
    base_loc_pred=None,
    base_loc_mask=None,
    base_loc_image_pred=None,
    base_loc_image_mask=None,
):
    model = trainer.model
    with torch.no_grad():
        inner_out = model(batch["edit_inner"])
        outer_out = model(batch["edit_outer"])
        image_out = model(batch["edit_outer_image"])

    inner_logits, inner_labels = _extract_logits_labels(inner_out, batch["edit_inner"])
    outer_logits, outer_labels = _extract_logits_labels(outer_out, batch["edit_outer"])
    image_logits, image_labels = _extract_logits_labels(image_out, batch["edit_outer_image"])

    inner_stats = model.edit_loss_fn(trainer.config, inner_logits, inner_labels, multimodal=True)
    outer_stats = model.edit_loss_fn(trainer.config, outer_logits, outer_labels, multimodal=True)
    image_stats = model.edit_loss_fn(trainer.config, image_logits, image_labels, multimodal=True)

    stats = {
        "inner/acc": inner_stats["acc"],
        "edit/acc": outer_stats["acc"],
        "image_rephrase/acc": image_stats["acc"],
    }
    if base_loc_topk is not None and base_loc_image_topk is not None:
        with torch.no_grad():
            loc_out = model(batch["loc"])
            loc_image_out = model(batch["loc_image"])
        post_loc_topk, post_loc_pred, _ = _locality_pack(loc_out, batch["loc"], 1)
        post_loc_image_topk, post_loc_image_pred, _ = _locality_pack(
            loc_image_out, batch["loc_image"], 10
        )
        stats["loc/acc"] = _consistency_acc(post_loc_topk, base_loc_topk)
        stats["image_loc/acc"] = _consistency_acc(post_loc_image_topk, base_loc_image_topk)
        if base_loc_pred is not None and base_loc_mask is not None:
            acc_answer = _masked_consistency_from_preds(
                post_loc_pred, base_loc_pred, base_loc_mask
            )
            if acc_answer is not None:
                stats["loc/acc_answer"] = acc_answer
        if base_loc_image_pred is not None and base_loc_image_mask is not None:
            acc_image_answer = _masked_consistency_from_preds(
                post_loc_image_pred, base_loc_image_pred, base_loc_image_mask
            )
            if acc_image_answer is not None:
                stats["image_loc/acc_answer"] = acc_image_answer
    return stats


def _eval_frozen(
    trainer,
    dataset,
    steps,
    base_loc_topks,
    base_loc_image_topks,
    base_loc_preds,
    base_loc_masks,
    base_loc_image_preds,
    base_loc_image_masks,
):
    max_steps = min(
        len(dataset),
        steps,
        len(base_loc_topks),
        len(base_loc_image_topks),
        len(base_loc_preds),
        len(base_loc_masks),
        len(base_loc_image_preds),
        len(base_loc_image_masks),
    )
    averager = RunningStatAverager()
    for idx in range(max_steps):
        batch = dataset.collate_fn([dataset[idx]])
        stats = _eval_batch_no_edit(
            trainer,
            batch,
            base_loc_topks[idx],
            base_loc_image_topks[idx],
            base_loc_preds[idx],
            base_loc_masks[idx],
            base_loc_image_preds[idx],
            base_loc_image_masks[idx],
        )
        averager.add(stats)
    summary = averager.average()
    summary["eval_edits"] = max_steps
    return summary


def _print_frozen(step: int, stats: Dict[str, Optional[float]], label: str = "Frozen Eval"):
    rel = _format_metric(stats, "inner/acc")
    t_gen = _format_metric(stats, "edit/acc")
    m_gen = _format_metric(stats, "image_rephrase/acc")
    t_loc = _format_metric(stats, "loc/acc")
    m_loc = _format_metric(stats, "image_loc/acc")
    t_loc_ans = _format_metric(stats, "loc/acc_answer")
    m_loc_ans = _format_metric(stats, "image_loc/acc_answer")
    print(
        f">>> [{label}] "
        f"{step} samples | "
        f"inner/acc={rel} "
        f"edit/acc={t_gen} "
        f"image_rephrase/acc={m_gen} "
        f"loc/acc={t_loc} "
        f"image_loc/acc={m_loc} "
        f"loc/acc_answer={t_loc_ans} "
        f"image_loc/acc_answer={m_loc_ans}"
    )

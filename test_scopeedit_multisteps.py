import argparse
import logging
import os
import time

import torch

from easyeditor import MOREMultimodalTrainingHparams, MultimodalTrainer, CaptionDataset, VQADataset
from easyeditor.trainer.utils import RunningStatAverager

from scopeedit_eval_utils import (
    _SliceDataset,
    _collect_locality_baseline,
    _eval_batch_no_edit,
    _eval_frozen,
    _freeze_model_params,
    _print_frozen,
    _print_lifelong,
    _print_lora_summary,
    _print_run_config,
)


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "hparams", "TRAINING", "MORE", "llava15_scopeedit.yaml")
# DEFAULT_TRAIN_DS = "/root/autodl-tmp/data/editing-data/vqa/vqa_train.json"
# DEFAULT_TEST_DS = "/root/autodl-tmp/data/editing-data/vqa/vqa_eval.json"

DEFAULT_TRAIN_DS = "/root/autodl-tmp/data/editing-data/caption/caption_train_edit.json"
DEFAULT_TEST_DS = "/root/autodl-tmp/data/editing-data/caption/caption_eval_edit.json"

# train_ds_path = "/root/autodl-tmp/data/editing-data/vqa/vqa_train.json"
# test_ds_path = "/root/autodl-tmp/data/editing-data/vqa/vqa_eval.json"
# train_ds = VQADataset(train_ds_path, config=hparams)
# test_ds = VQADataset(test_ds_path, config=hparams)


def _parse_edit_steps(raw):
    steps = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            steps.append(int(part))
    return steps or [1, 10, 100, 1000]


def _resolve_dataset_cls(config_path, train_ds_path, test_ds_path):
    marker = " ".join([str(config_path), str(train_ds_path), str(test_ds_path)]).lower()
    if "vqa" in marker:
        return VQADataset
    return CaptionDataset


def _run_more_steps(trainer, batch, more_steps):
    model = trainer.model
    model.train(False)
    trainer.original_model.train(False)

    total_time = 0.0
    last_info = {}
    active_group_union = []
    for _ in range(max(1, int(more_steps))):
        model.zero_grad(set_to_none=True)
        start = time.time()
        _, step_info = model.edit(batch["edit_inner"], batch["cond"])
        total_time += time.time() - start

        if isinstance(step_info, dict):
            step_info = dict(step_info)
            restore_fn = step_info.pop("restore_fn", None)
            if restore_fn is not None:
                restore_fn()
            for group in step_info.get("more/top_groups", []) or []:
                if group not in active_group_union:
                    active_group_union.append(group)
            last_info = step_info

    model.zero_grad(set_to_none=True)
    model.eval()

    last_info["time/edit"] = total_time
    last_info["more/steps"] = max(1, int(more_steps))
    return last_info, active_group_union


def _merge_numeric_info(dst, src):
    if not isinstance(src, dict):
        return
    for key, value in src.items():
        if isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0.0) + float(value)
        else:
            dst[key] = value


def _accumulate_locality_keys(trainer, batch, active_groups):
    model = trainer.model
    accum_fn = getattr(model, "accumulate_current_keys", None)
    if not callable(accum_fn):
        return {}
    if not bool(getattr(model.config, "more_accumulate_locality_keys", False)):
        return {}

    stats = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(batch["loc"])
    _merge_numeric_info(stats, accum_fn(active_groups, source="loc"))
    with torch.no_grad():
        model(batch["loc_image"])
    _merge_numeric_info(stats, accum_fn(active_groups, source="loc_image"))
    model.train(was_training)
    return stats


def run_continuous_eval(
    config_path=DEFAULT_CONFIG,
    train_ds_path=DEFAULT_TRAIN_DS,
    test_ds_path=DEFAULT_TEST_DS,
    warmup_iters=100,
    n_edits=100,
    edit_steps=None,
    more_steps=1,
):
    if edit_steps is None:
        edit_steps = [1, 10, 100, 1000]

    hparams = MOREMultimodalTrainingHparams.from_hparams(config_path)
    if warmup_iters > 0:
        hparams.eval_only = False
        hparams.final_eval = False
        hparams.max_iters = warmup_iters
        hparams.max_epochs = None
        hparams.val_interval = warmup_iters + 1
    else:
        hparams.eval_only = True
        hparams.final_eval = False

    _print_run_config("Continuous", config_path, hparams, warmup_iters, n_edits, edit_steps)
    print(f">>> [Continuous] more_steps={max(1, int(more_steps))} (continuous eval only)")
    print(f">>> [Continuous] Loading Train Dataset from: {train_ds_path}")
    print(f">>> [Continuous] Loading Test Dataset from: {test_ds_path}")
    dataset_cls = _resolve_dataset_cls(config_path, train_ds_path, test_ds_path)
    train_ds = dataset_cls(train_ds_path, config=hparams)
    test_ds = dataset_cls(test_ds_path, config=hparams)

    max_edits = min(len(test_ds), n_edits)
    eval_ds = _SliceDataset(test_ds, max_edits)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds,
    )
    _print_lora_summary("Continuous", trainer)

    if warmup_iters > 0:
        trainer.model.config.more_update_mode = "online"
        trainer.model.config.more_restore_p = False
        print(f">>> [Warmup] Starting M-ORE training for {warmup_iters} steps...")
        trainer.run()
        print(">>> [Warmup] Done")

    trainer.model.config.more_update_mode = "online"
    trainer.model.config.more_restore_p = False

    edit_steps = [s for s in edit_steps if s <= max_edits]
    records = [eval_ds[i] for i in range(max_edits)]

    print(f">>> [Continuous] Starting sequential edits for {max_edits} samples...")
    base_loc_topks = []
    base_loc_image_topks = []
    base_loc_preds = []
    base_loc_masks = []
    base_loc_image_preds = []
    base_loc_image_masks = []
    averager = RunningStatAverager("val")
    online_results = {}
    frozen_step_results = {}

    for idx, record in enumerate(records, start=1):
        batch = eval_ds.collate_fn([record])
        (
            loc_topk,
            loc_pred,
            loc_mask,
            loc_image_topk,
            loc_image_pred,
            loc_image_mask,
        ) = _collect_locality_baseline(trainer, batch)
        base_loc_topks.append(loc_topk)
        base_loc_image_topks.append(loc_image_topk)
        base_loc_preds.append(loc_pred)
        base_loc_masks.append(loc_mask)
        base_loc_image_preds.append(loc_image_pred)
        base_loc_image_masks.append(loc_image_mask)

        edit_info, active_groups = _run_more_steps(trainer, batch, more_steps)
        info_dict = _eval_batch_no_edit(
            trainer,
            batch,
            loc_topk,
            loc_image_topk,
            loc_pred,
            loc_mask,
            loc_image_pred,
            loc_image_mask,
        )
        _merge_numeric_info(
            info_dict,
            _accumulate_locality_keys(trainer, batch, active_groups),
        )
        info_dict.update(edit_info)
        averager.add(info_dict)

        if idx in edit_steps:
            stats = averager.average()
            stats["eval_edits"] = idx
            online_results[idx] = stats
            _print_lifelong(idx, stats)
            frozen_stats = _eval_frozen(
                trainer,
                eval_ds,
                idx,
                base_loc_topks,
                base_loc_image_topks,
                base_loc_preds,
                base_loc_masks,
                base_loc_image_preds,
                base_loc_image_masks,
            )
            frozen_step_results[idx] = frozen_stats
            _print_frozen(idx, frozen_stats, label="Frozen Eval")

    print(">>> [Continuous] Freezing parameters and evaluating edited samples...")
    _freeze_model_params(trainer.model)
    frozen_stats = frozen_step_results[max(frozen_step_results)] if frozen_step_results else {}
    return {
        "online": online_results,
        "frozen_steps": frozen_step_results,
        "frozen": frozen_stats,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Continuous M-ORE / ScopeEdit eval with more_steps")
    parser.add_argument("config", nargs="?", default=os.environ.get("MORE_CONFIG_PATH", DEFAULT_CONFIG))
    parser.add_argument("--train-ds", default=DEFAULT_TRAIN_DS)
    parser.add_argument("--test-ds", default=DEFAULT_TEST_DS)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--n-edits", type=int, default=100)
    parser.add_argument("--edit-steps", default="1,10,20,30,100,1000")
    parser.add_argument("--more-steps", type=int, default=int(os.environ.get("MORE_STEPS", "5")))
    return parser.parse_args()


def main():
    args = _parse_args()
    run_continuous_eval(
        config_path=args.config,
        train_ds_path=args.train_ds,
        test_ds_path=args.test_ds,
        warmup_iters=args.warmup_iters,
        n_edits=args.n_edits,
        edit_steps=_parse_edit_steps(args.edit_steps),
        more_steps=args.more_steps,
    )


if __name__ == "__main__":
    main()

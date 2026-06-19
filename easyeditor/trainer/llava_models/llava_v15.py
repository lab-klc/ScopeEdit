from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration

from ...util.lora_layers import LoRALinear


@dataclass
class LlavaOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    labels: Optional[torch.IntTensor] = None
    attention_mask: Optional[torch.IntTensor] = None


class _SimpleProcessor:
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, return_tensors="pt", padding=True):
        if text is None:
            raise ValueError("LLaVA processor requires text inputs.")
        tokens = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=True
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
            tokens["pixel_values"] = image_inputs["pixel_values"]
        return tokens


class LlavaV15ForEditing(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float32,
        device_map: Optional[str] = None,
        use_chat_template: bool = True,
        image_processor_name_or_path: Optional[str] = None,
        scopeedit_visual_anchor_enable: bool = False,
        scopeedit_visual_anchor_start: Optional[int] = None,
        scopeedit_visual_anchor_end: Optional[int] = None,
        bridge_visual_anchor_enable: Optional[bool] = None,
        bridge_visual_anchor_start: Optional[int] = None,
        bridge_visual_anchor_end: Optional[int] = None,
    ):
        super().__init__()
        if bridge_visual_anchor_enable is not None:
            scopeedit_visual_anchor_enable = bridge_visual_anchor_enable
        if bridge_visual_anchor_start is not None:
            scopeedit_visual_anchor_start = bridge_visual_anchor_start
        if bridge_visual_anchor_end is not None:
            scopeedit_visual_anchor_end = bridge_visual_anchor_end

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True, use_fast=False
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            image_proc_path = image_processor_name_or_path or "openai/clip-vit-large-patch14-336"
            image_processor = CLIPImageProcessor.from_pretrained(image_proc_path)
            self.processor = _SimpleProcessor(image_processor, tokenizer)
            self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
        self.config = self.model.config
        self.use_chat_template = use_chat_template
        self.scopeedit_visual_anchor_enable = bool(scopeedit_visual_anchor_enable)
        self.scopeedit_visual_anchor_start = (
            None if scopeedit_visual_anchor_start is None else int(scopeedit_visual_anchor_start)
        )
        self.scopeedit_visual_anchor_end = (
            None if scopeedit_visual_anchor_end is None else int(scopeedit_visual_anchor_end)
        )

        # Alias attributes to satisfy M-ORE injection paths.
        self.llama_model = getattr(self.model, "language_model", None)
        if self.llama_model is None:
            self.llama_model = getattr(self.model, "model", None)

        self.llama_proj = None
        for attr in ("multi_modal_projector", "mm_projector"):
            if hasattr(self.model, attr):
                self.llama_proj = getattr(self.model, attr)
                break

        if self.llama_model is None or self.llama_proj is None:
            raise ValueError(
                "LLaVA model must expose language_model/model and a multimodal projector."
            )

        # Used by M-ORE target-length masking.
        self.llama_tokenizer = self.tokenizer

    def _resolve_scopeedit_visual_anchor_range(self):
        if not self.scopeedit_visual_anchor_enable:
            return None, None

        layers = None
        if hasattr(self.llama_model, "model") and hasattr(self.llama_model.model, "layers"):
            layers = self.llama_model.model.layers
        elif hasattr(self.llama_model, "layers"):
            layers = self.llama_model.layers
        if layers is None:
            return None, None

        n_layers = len(layers)
        if (
            self.scopeedit_visual_anchor_start is None
            and self.scopeedit_visual_anchor_end is None
        ):
            return None, None

        start = (
            self.scopeedit_visual_anchor_start
            if self.scopeedit_visual_anchor_start is not None
            else self.scopeedit_visual_anchor_end
        )
        end = (
            self.scopeedit_visual_anchor_end
            if self.scopeedit_visual_anchor_end is not None
            else self.scopeedit_visual_anchor_start
        )
        start = max(0, min(int(start), n_layers - 1))
        end = max(0, min(int(end), n_layers - 1))
        if start > end:
            start, end = end, start
        return start, end

    def _resolve_image_token_id(self):
        token_id = getattr(self.config, "image_token_index", None)
        if token_id is not None:
            return int(token_id)

        image_token = getattr(self.processor, "image_token", None)
        if image_token is None:
            image_token = getattr(self.tokenizer, "image_token", None)
        if image_token is None and hasattr(self.tokenizer, "additional_special_tokens"):
            for token in self.tokenizer.additional_special_tokens:
                if "image" in token.lower():
                    image_token = token
                    break
        if image_token is None or not hasattr(self.tokenizer, "convert_tokens_to_ids"):
            return None

        token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            return None
        return int(token_id)

    def _build_scopeedit_context(self, inputs):
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return None

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.bool()

        image_token_id = self._resolve_image_token_id()
        if image_token_id is None:
            image_mask = torch.zeros_like(attention_mask)
        else:
            image_mask = input_ids == image_token_id

        text_mask = attention_mask & ~image_mask
        text_reference = None
        embedder = None
        if hasattr(self.model, "get_input_embeddings"):
            embedder = self.model.get_input_embeddings()
        elif hasattr(self.llama_model, "get_input_embeddings"):
            embedder = self.llama_model.get_input_embeddings()
        if embedder is not None:
            text_reference = embedder(input_ids)

        anchor_start, anchor_end = self._resolve_scopeedit_visual_anchor_range()

        return {
            "text_mask": text_mask,
            "image_mask": image_mask,
            "text_reference": text_reference,
            "text_reference_mask": text_mask,
            "visual_anchor_enable": self.scopeedit_visual_anchor_enable,
            "visual_anchor_start": anchor_start,
            "visual_anchor_end": anchor_end,
        }

    def _build_chat_prompt(self, prompt: str, has_image: bool) -> str:
        if not self.use_chat_template:
            return prompt
        if has_image:
            content = [{"type": "image"}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]
        template_fn = getattr(self.processor, "apply_chat_template", None)
        if template_fn is None:
            template_fn = getattr(self.tokenizer, "apply_chat_template", None)
        if template_fn is None:
            raise ValueError("No chat template available for LLaVA processor/tokenizer.")
        return template_fn(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def compute_more_target_lens(self, batch) -> Optional[List[int]]:
        targets = batch.get("target_text", None)
        if targets is None:
            return None
        if isinstance(targets, torch.Tensor):
            return [int(targets.size(1)) for _ in range(targets.size(0))]
        targets = self._normalize_targets(targets)
        return [
            len(self.tokenizer.encode(t, add_special_tokens=False)) for t in targets
        ]

    @staticmethod
    def _normalize_targets(targets: List[str]) -> List[str]:
        normalized = []
        for t in targets:
            if t and not t.startswith(" "):
                normalized.append(" " + t)
            else:
                normalized.append(t)
        return normalized

    def _build_labels(self, input_ids: torch.Tensor, targets: List[str]) -> torch.Tensor:
        labels = torch.full_like(input_ids, -100)
        if not targets:
            return labels

        tokenized = self.tokenizer(
            targets, add_special_tokens=False, return_tensors="pt", padding=True
        )
        target_ids = tokenized["input_ids"].to(input_ids.device)
        pad_id = self.tokenizer.pad_token_id

        for i in range(input_ids.size(0)):
            tlen = int((target_ids[i] != pad_id).sum())
            if tlen > 0:
                labels[i, -tlen:] = target_ids[i, :tlen]
        return labels

    def forward(self, samples):
        prompts = samples.get("prompt_text", None)
        targets = samples.get("target_text", None)
        text_inputs = samples.get("text_input", None)

        images = samples.get("image", None)
        has_image = images is not None

        if prompts is not None and targets is not None:
            targets = self._normalize_targets(targets)
            chat_prompts = [self._build_chat_prompt(p, has_image) for p in prompts]
            full_texts = [p + t for p, t in zip(chat_prompts, targets)]
        elif text_inputs is not None:
            full_texts = text_inputs
        else:
            raise ValueError("Missing prompt/target or text_input in batch.")

        if has_image:
            inputs = self.processor(
                images=images, text=full_texts, return_tensors="pt", padding=True
            )
        else:
            inputs = self.processor(
                text=full_texts, return_tensors="pt", padding=True
            )

        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if targets is not None:
            labels = self._build_labels(inputs["input_ids"], targets)
        else:
            labels = torch.full_like(inputs["input_ids"], -100)
        scopeedit_state = LoRALinear.stash_scopeedit_context()
        scopeedit_context = self._build_scopeedit_context(inputs)
        if scopeedit_context is not None:
            LoRALinear.set_scopeedit_context(**scopeedit_context)
        try:
            outputs = self.model(**inputs, labels=labels, return_dict=True)
        finally:
            LoRALinear.restore_scopeedit_context(scopeedit_state)

        return LlavaOutput(
            loss=outputs.loss if hasattr(outputs, "loss") else None,
            logits=outputs.logits,
            labels=labels,
            attention_mask=inputs.get("attention_mask"),
        )

"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from .blip2 import Blip2Base, disabled_train
from .modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

from ...util.lora_layers import LoRALinear

@dataclass
class BLIP2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.IntTensor = None
    attention_mask: torch.IntTensor = None


class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        opt_precision="fp16",
        prompt="",
        max_txt_len=2048,
        state_dict_file=None,
        qformer_name_or_path="bert-base-uncased",
        qformer_checkpoint="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt2.7b.pth"
    ):
        super().__init__()
        self.config = None
        self.tokenizer = self.init_tokenizer(qformer_name_or_path)

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, state_dict_file
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, qformer_name_or_path
        ) # query_token?
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        opt_precision = (opt_precision or "fp16").lower()
        if opt_precision in ("bf16", "bfloat16"):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                opt_dtype = torch.bfloat16
            else:
                logging.warning("BF16 not supported; falling back to FP16 for OPT.")
                opt_dtype = torch.float16
        elif opt_precision in ("fp32", "float32"):
            opt_dtype = torch.float32
        else:
            opt_dtype = torch.float16
        self.opt_precision = opt_precision
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.vision_qformer_autocast_dtype = torch.bfloat16
        else:
            logging.warning(
                "BF16 not supported; vision/Q-Former autocast will use FP16."
            )
            self.vision_qformer_autocast_dtype = torch.float16

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        if self.opt_tokenizer.pad_token is None:
            self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token
        self.llama_tokenizer = self.opt_tokenizer
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=opt_dtype
        )
        # for name, param in self.opt_model.named_parameters():
        #     param.requires_grad = False
        # self.eos_token_id = self.opt_tokenizer(
        #     "\n", add_special_tokens=False
        # ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        
        print('Loading Q-Former and Linear')
        self.load_from_pretrained(url_or_filename=qformer_checkpoint)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former and Linear Done')
        
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def _build_scopeedit_context(
        self,
        attention_mask: Optional[torch.Tensor],
        text_reference: Optional[torch.Tensor],
        text_reference_mask: Optional[torch.Tensor],
        num_image_tokens: int,
    ):
        if attention_mask is None:
            return None

        attention_mask = attention_mask.bool()
        image_mask = torch.zeros_like(attention_mask)
        if num_image_tokens > 0:
            prefix_len = min(int(num_image_tokens), attention_mask.size(1))
            image_mask[:, :prefix_len] = attention_mask[:, :prefix_len]
        text_mask = attention_mask & ~image_mask

        if text_reference_mask is not None:
            text_reference_mask = text_reference_mask.bool()

        return {
            "text_mask": text_mask,
            "image_mask": image_mask,
            "text_reference": text_reference,
            "text_reference_mask": text_reference_mask,
            "visual_anchor_enable": False,
            "visual_anchor_start": None,
            "visual_anchor_end": None,
        }

    def forward(self, samples):
        text = [t for t in samples["text_input"]]
        prompts_len = samples.get("prompts_len")
        has_image = samples["image"] is not None
        device = samples["image"].device if has_image else self.opt_model.device

        self.opt_tokenizer.padding_side = "right"
        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            # max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if prompts_len is not None:
            for i, prompt_len in enumerate(prompts_len):
                targets[i, :prompt_len] = -100

        text_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        text_reference_mask = opt_tokens.attention_mask.bool()

        scopeedit_state = LoRALinear.stash_scopeedit_context()
        LoRALinear.set_scopeedit_context(
            text_reference=text_embeds,
            text_reference_mask=text_reference_mask,
        )
        try:
            if has_image:
                image = samples["image"]  # bsz, 3, image_size, image_size
                with self.maybe_autocast(dtype=self.vision_qformer_autocast_dtype):
                    image_embeds = self.ln_vision(self.visual_encoder(image))
                    image_atts = torch.ones(
                        image_embeds.size()[:-1], dtype=torch.long, device=image.device
                    )

                    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )

                    # Project query-conditioned visual tokens into OPT space.
                    inputs_opt = self.opt_proj(query_output.last_hidden_state)
                atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long, device=image.device)

                empty_targets = torch.full_like(atts_opt, -100)
                targets = torch.cat([empty_targets, targets], dim=1)

                inputs_opt = inputs_opt.to(text_embeds.dtype)
                inputs_embeds = torch.cat([inputs_opt, text_embeds], dim=1)
                attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
                num_image_tokens = atts_opt.size(1)
            else:
                inputs_embeds = text_embeds
                attention_mask = opt_tokens.attention_mask
                num_image_tokens = 0

            scopeedit_context = self._build_scopeedit_context(
                attention_mask=attention_mask,
                text_reference=text_embeds,
                text_reference_mask=text_reference_mask,
                num_image_tokens=num_image_tokens,
            )
            if scopeedit_context is not None:
                LoRALinear.set_scopeedit_context(**scopeedit_context)

            opt_dtype = self.opt_model.dtype
            autocast_dtype = (
                opt_dtype if opt_dtype in (torch.float16, torch.bfloat16) else torch.float32
            )
            with self.maybe_autocast(dtype=autocast_dtype):
                outputs = self.opt_model(
                    inputs_embeds=inputs_embeds,  # image prefix + text token embeddings
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        finally:
            LoRALinear.restore_scopeedit_context(scopeedit_state)
        loss = outputs.loss

        if torch.isnan(outputs.logits).any():
            print("NAN in logits!!!")

        return BLIP2Output(
            loss=loss,
            logits=outputs.logits,
            labels=targets,
            attention_mask=attention_mask
        )
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with self.maybe_autocast(dtype=self.vision_qformer_autocast_dtype):
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            # inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            # inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            # outputs = self.opt_model.generate(
            #     inputs_embeds=inputs_embeds, 
            #     attention_mask=attention_mask,
            #     do_sample=use_nucleus_sampling,
            #     top_p=top_p,
            #     temperature=temperature,
            #     num_beams=num_beams,
            #     max_length=max_length,
            #     min_length=min_length,
            #     eos_token_id=self.eos_token_id,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     num_return_sequences=num_captions,
            # )
            # output_text = self.opt_tokenizer.batch_decode(
            #     outputs, skip_special_tokens=True
            # )
                            
            # previous version for transformers<4.27
            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            outputs = self.opt_model.generate(
                input_ids=opt_tokens.input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            
            output_text = [text.strip() for text in output_text]
            return output_text

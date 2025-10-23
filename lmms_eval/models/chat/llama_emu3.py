import base64
import re
import time
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    LlamaForCausalLM
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.protocol import ChatMessages

import sys
import os

# refer to swissai specific repositories on clariden
user = os.environ["USER"]
tokenizer_path = f"/iopsstor/scratch/cscs/{user}/benchmark-image-tokenzier"
sys.path.append(tokenizer_path)
from vision_tokenization.utils.tokenization_emu3_image_only import EMU3ImageOnlyTokenizer


@register_model("llama_emu3")
class LlamaEmu3Chat(lmms):
    """
    Integration of a llama3.2-3B model with a unified multi-modal early fusion approach. Model Vocabulary is extended to include image tokens.
    Emu3 tokenizer is used to convert images to discrete tokens.

    Part of the code is based on chat/qwen2_5_vl.py
    """
    is_simple = False

    def __init__(
        self,
        pretrained: str = "/iopsstor/scratch/cscs/nirmiger/Megatron-LM/logs/Meg-Runs/image-extension/llama3-3b-2n-8192sl-120gbsz-0.5-0.5/HF",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        reasoning_prompt: Optional[str] = None,
        tokenizer_path: Optional[str] = "/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer",
        emu3_min_pixels: Optional[int] = 512 * 512,
        emu3_max_pixels: Optional[int] = 1024 * 1024,
        max_length: Optional[int] = None,
        ignore_max_length: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Check for unexpected kwargs
        if kwargs:
            eval_logger.warning(f"Unexpected kwargs will be ignored: {kwargs}")

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = LlamaForCausalLM.from_pretrained(pretrained, **model_kwargs).eval()

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.image_tokenizer = EMU3ImageOnlyTokenizer(
            text_tokenizer_path=tokenizer_path,
            device=self._device,
            min_pixels=emu3_min_pixels,
            max_pixels=emu3_max_pixels,
        )
        self.image_tokenizer.to(self._device) # TODO: add image tokenizer to accelerator?
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.system_prompt = system_prompt

        self._config = self.model.config

        # Set max_length with sensible defaults
        if max_length is not None:
            self._max_length = max_length
        else:
            # Try to get from model config, fallback to 8192 (Llama3B default)
            try:
                self._max_length = self._config.max_position_embeddings
                eval_logger.info(f"Using max_length from model config: {self._max_length}")
            except AttributeError:
                self._max_length = 8192
                eval_logger.warning(f"Could not infer max_length from model config, using default: {self._max_length}")

        # Store flag to optionally ignore max_length (for testing beyond model capacity)
        self.ignore_max_length = ignore_max_length
        if self.ignore_max_length:
            eval_logger.warning("ignore_max_length=True: Truncation disabled. Long sequences may cause OOM or errors.")

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.sft_eot_token

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for llama_emu3 model.")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[2], x[2]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[0](self.task_dict[task][split][ids]) for ids, task, split in zip(doc_id, task, split)]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]

            gen_kwargs = all_gen_kwargs[0]

            # TODO: Here i should apply a chat template, but do not have one at the moment
            batched_messages = [chat_message.to_hf_messages() for chat_message in chat_messages]

            # extract images from messages
            images = []
            for messages in chat_messages:
                image, _, _ = messages.extract_media()
                images.append(image)
            images = self.flatten(images)

            # Run Emu3 tokenizer to get discrete image tokens
            image_token_strs = []
            for img in images:
                with torch.inference_mode():
                    # PIL IMG to emu3 discrete text token ids
                    emu_text = self.image_tokenizer.translate_image_to_text(img)  # returns list of token IDs
                # convert token IDs to special token strings known to text tokenizer
                image_token_strs.append(emu_text)

            # TODO: is the format of messages correctly anticipated?
            texts = []
            # For each msg, create a string of text inputs and image placeholders
            for msgs in batched_messages:
                # Extract text only from each message
                msg_text = []
                for m in msgs:
                    for c in m["content"]:
                        if c["type"] == "text":
                            msg_text.append(c["text"])
                        elif c["type"] == "image":
                            # insert a placeholder to replace with image tokens
                            msg_text.append("<image>")
                texts.append(" ".join(msg_text))

            # Replace <image> markers with actual image token strings
            # Track which image belongs to which text
            image_idx = 0
            processed_texts = []
            for txt in texts:
                num_images = txt.count("<image>")
                for _ in range(num_images):
                    if image_idx < len(image_token_strs):
                        txt = txt.replace("<image>", image_token_strs[image_idx], 1)
                        image_idx += 1
                processed_texts.append(txt)

            # Tokenize to get input ids with truncation if not ignoring max_length
            if self.ignore_max_length:
                # No truncation - allow sequences to exceed max_length (may cause errors)
                inputs = self._tokenizer(processed_texts, padding=True, return_tensors="pt")
            else:
                # First, check actual lengths to log truncation statistics
                untruncated_inputs = self._tokenizer(processed_texts, padding=False, return_tensors=None)
                actual_lengths = [len(ids) for ids in untruncated_inputs["input_ids"]]
                exceeded_count = sum(1 for length in actual_lengths if length > self._max_length)

                if exceeded_count > 0:
                    max_actual_length = max(actual_lengths)
                    avg_exceeded_length = sum(length for length in actual_lengths if length > self._max_length) / exceeded_count
                    eval_logger.warning(
                        f"Truncation: {exceeded_count}/{len(actual_lengths)} sequences exceed max_length ({self._max_length}). "
                        f"Max length: {max_actual_length}, Avg exceeded length: {avg_exceeded_length:.0f}"
                    )

                # Now tokenize with truncation
                inputs = self._tokenizer(processed_texts, padding=True, max_length=self._max_length, truncation=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            # TODO: should be able to set stop token ids or set them explicitly here?
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            # Generate responses
            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            end_time = time.time()

            # Decode responses
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self._tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

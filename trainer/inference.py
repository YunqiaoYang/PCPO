import json
from difflib import SequenceMatcher
import torch
import yaml
import warnings
import math
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    is_wandb_available,
)
from trl.models import PreTrainedModelWrapper
from trl.trainer.utils import (
    RunningMoments,
    cap_exp,
    disable_dropout_in_model,
    generate_model_card,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16,
)
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group("nccl")
world_size = dist.get_world_size()
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

def save_jsonl(data, path,mode='w',**kwargs):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False,**kwargs) + '\n')

class InferenceDataset(Dataset):
    """
    A simple dataset class for inference.
    Assumes the input is already tokenized and contains the required fields.
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class DPOInference:
    def __init__(
        self,
        model,
        model_init_kwargs,
        tokenizer,
        args,
        beta=0.1,
        label_pad_token_id=-100,
        max_prompt_length=None,
        max_completion_length=None,
        padding_value: Optional[int] = None
    ):  
        self.args=args
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the DPOTrainer/DPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
            
        if label_pad_token_id != -100:
            warnings.warn(
                "You passed `label_pad_token_id` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_pad_token_id = label_pad_token_id

        if padding_value is not None:
            warnings.warn(
                "You passed `padding_value` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.padding_value = padding_value
        processing_class=tokenizer
        if args.padding_value is not None:
            self.padding_value = args.padding_value
        else:
            if hasattr(processing_class, "pad_token_id") and processing_class.pad_token_id is not None:
                self.padding_value = processing_class.pad_token_id
            elif hasattr(processing_class, "tokenizer") and processing_class.tokenizer.pad_token_id is not None:
                self.padding_value = processing_class.tokenizer.pad_token_id
            else:
                raise ValueError(
                    "Can't find `pad_token_id` in the `processing_class`. "
                    "Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`) "
                    "before instantiating the trainer."
                )
        
        
        self.device = local_rank
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.label_pad_token_id = label_pad_token_id
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

        
        self.model.eval()

    
    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        """
        Tokenize a row of the dataset.

        Args:
            features (`Dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
            processing_class (`PreTrainedTokenizerBase`):
                Processing class used to process the data.
            max_prompt_length (`int` or `None`):
                Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            add_special_tokens (`bool`):
                Whether to add special tokens to the sequences. Typically used for encoder-decoder models. If `True`,
                the prompt sequence will have a bos token prepended and an eos token appended. In any case, the
                completion sequences will have an eos token appended.

        Returns:
            `Dict[str, List[int]]`:
                Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
                `"rejected_input_ids".

        Example:
        ```python
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
        >>> DPOTrainer.tokenize_row(features, tokenizer, max_prompt_length=3, max_completion_length=3, add_special_tokens=False)
        {'prompt_input_ids': [464, 6766, 318], 'chosen_input_ids': [4171, 50256], 'rejected_input_ids': [4077, 50256]}
        ```
        """
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]], padding_value: int
    ) -> Dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt
        and completion sequences.

        Args:
            batch (`Dict[str, Union[List, torch.LongTensor]]`):
                A batch of input data. The batch must contain the following keys:

                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected completion input IDs.
                - `"prompt_pixel_values"` (optional): Tensor for pixel values, if available.
                - `"prompt_pixel_attention_mask"` (optional): Tensor for pixel attention masks, if available.

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences (`chosen_input_ids` and
                `rejected_input_ids`).

        Returns:
            `Dict[str, torch.LongTensor]`: A dictionary containing:

                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen and rejected completion input IDs of shape `(2 * batch_size, max_completion_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size, prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen and rejected attention masks of shape `(2 * batch_size, max_completion_length)`.
                - `"pixel_values"` (optional): Concatenated pixel values if `"prompt_pixel_values"` are present.
                - `"pixel_attention_mask"` (optional): Concatenated pixel attention masks if `"prompt_pixel_attention_mask"` are present.

        Notes:
            The completion input IDs and attention masks are padded to the maximum completion length of the chosen
            or rejected sequences.
        """
        output = {}

        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        if "pixel_values" in batch:
            output["pixel_values"] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)

        if "pixel_attention_mask" in batch:
            output["pixel_attention_mask"] = torch.cat(
                [batch["pixel_attention_mask"], batch["pixel_attention_mask"]], dim=0
            )

        # Concatenate the chosen and rejected completions
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        return output
    
    @torch.no_grad()
    def _concatenated_forward(self, batch):
        """
        Perform a forward pass to compute per-token log probabilities
        for chosen and rejected sequences in a single forward operation.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            for i in range(attention_mask.size(0)):
                first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

            # Get the first column idx that is all zeros and remove every column after that
            empty_cols = torch.sum(attention_mask, dim=0) == 0
            first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1) + 1
            input_ids = input_ids[:, : first_empty_col - 1]
            attention_mask = attention_mask[:, : first_empty_col - 1]
            loss_mask = loss_mask[:, : first_empty_col - 1]

            # Truncate right
            if self.args.max_length is not None:
                input_ids = input_ids[:, : self.args.max_length]
                attention_mask = attention_mask[:, : self.args.max_length]
                loss_mask = loss_mask[:, : self.args.max_length]
            ##Flush over
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)

            # Offset the logits by one to align with the labels
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:].clone()
            loss_mask = loss_mask[:, 1:].bool()

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps[~loss_mask] = 0
        all_logps = per_token_logps.sum(-1)
        
        size_completion=loss_mask.sum(-1)
        chosen_per_token_logps, rejected_per_token_logps = per_token_logps.split(num_examples, dim=0)
        chosen_length, rejected_length = size_completion.split(num_examples, dim=0)


        return chosen_per_token_logps, rejected_per_token_logps, all_logps, chosen_length, rejected_length

    def _pad_to_length(self, tensor, length, pad_value):
        """
        Pad a tensor along dimension 1 to `length` with `pad_value`.
        """
        if tensor.size(1) < length:
            pad_size = length - tensor.size(1)
            pad_tensor = torch.full((tensor.size(0), pad_size), pad_value, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=1)
        return tensor

    @torch.no_grad()
    def compute_s_t(self, batch):
        """
        Compute s_t values for the sequences in a batch.

        Uses SequenceMatcher to find common subsequence tokens and compute s_t only on those tokens.
        """
        chosen_token_logps, rejected_token_logps, all_logps, chosen_length, rejected_length = self._concatenated_forward(batch)

        results = []
        for i in range(len(batch["prompt"])):
            chosen_ids = batch["chosen_input_ids"][i].tolist()
            rejected_ids = batch["rejected_input_ids"][i].tolist()

            chosen_ids = chosen_ids[:chosen_length[i]]
            rejected_ids = rejected_ids[:rejected_length[i]]
            chosen_mask, rejected_mask, index_mapping = self._compute_common_mask(chosen_ids, rejected_ids)

            filtered_chosen_ids = [chosen_ids[i] for i in range(len(chosen_mask)) if chosen_mask[i]]
            # filtered_rejected_ids = [rejected_ids[i] for i in range(len(rejected_mask)) if rejected_mask[i]]
            
            assert len(filtered_chosen_ids)==len(index_mapping)
            
            filtered_chosen = self.tokenizer.decode(filtered_chosen_ids,skip_special_tokens=False)

            st_values = []
            # Simplified approach: assume a one-to-one match in order
            for (idx_c, idx_r) in index_mapping:
                chosen_lp = chosen_token_logps[i][idx_c]
                rejected_lp = rejected_token_logps[i][idx_r]
                
                # s_t = -¦Â * chosen_lp + ¦Â * rejected_lp
                # cross entropy between chosen and rejected
                # >0 denotes rejected higer probability
                # <0 denotes chosen higer probability
                st_val = (-self.args.scale_alpha * chosen_lp + self.args.scale_alpha * rejected_lp).item()
                
                st_values.append(math.exp(-abs(st_val)))

            res = {
                "idx": batch["idx"][i],
                "prompt": batch["prompt"][i],
                "chosen": batch["chosen"][i],
                "rejected": batch["rejected"][i],
                "distance": batch["distance"][i],
                "nearest": batch["nearest"][i],
                "s_t_values": st_values,
                "s_t_values_all": sum(st_values),
                "s_t_values_weighted": sum(st_values)/rejected_length[i].cpu().numpy(),
                "index_mapping": index_mapping,
                "filtered_chosen": filtered_chosen
            }
            results.append(res)
            if self.args.save_ids:
                ids = {
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                    "filtered_chosen_ids": filtered_chosen_ids
                }
                results.append(ids)

        return results

    def _compute_common_mask(self, chosen_ids, rejected_ids, min_length=1):
        """
        Use SequenceMatcher to find the common subsequence.
        Returns masks for chosen and rejected sequences indicating common tokens.
        """
        matcher = SequenceMatcher(None, chosen_ids, rejected_ids)
        chosen_mask = [False]*len(chosen_ids)
        rejected_mask = [False]*len(rejected_ids)
        index_mapping = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal' and (i2 - i1) >= min_length:
                for ci, rj in zip(range(i1, i2), range(j1, j2)):
                    chosen_mask[ci] = True
                    rejected_mask[rj] = True
                    index_mapping.append((ci, rj))
                
        return chosen_mask, rejected_mask, index_mapping


    def _collate_fn(self, examples):
        """
        DataLoader collate function to prepare a batch.
        """
        prompt = [ex["origin_prompt"] for ex in examples]
        chosen = [ex["origin_chosen"] for ex in examples]
        rejected = [ex["origin_rejected"] for ex in examples]
        distance = [ex["distance"] for ex in examples]
        nearest = [ex["nearest"] for ex in examples]
        idx = [ex["idx"] for ex in examples]

        prompt_input_ids = [torch.tensor(ex["prompt_input_ids"]) for ex in examples]
        prompt_attention_mask = [torch.ones_like(ids) for ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(ex["chosen_input_ids"]) for ex in examples]
        chosen_attention_mask = [torch.ones_like(ids) for ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(ex["rejected_input_ids"]) for ex in examples]
        rejected_attention_mask = [torch.ones_like(ids) for ids in rejected_input_ids]

        prompt_input_ids = pad(prompt_input_ids, padding_value=self.padding_value, padding_side="left")
        prompt_attention_mask = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        chosen_input_ids = pad(chosen_input_ids, padding_value=self.padding_value)
        chosen_attention_mask = pad(chosen_attention_mask, padding_value=0)
        rejected_input_ids = pad(rejected_input_ids, padding_value=self.padding_value)
        rejected_attention_mask = pad(rejected_attention_mask, padding_value=0)

        return {
            "idx": idx,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "distance": distance,
            "nearest": nearest,
            "prompt_input_ids": prompt_input_ids.to(self.device),
            "prompt_attention_mask":prompt_attention_mask.to(self.device),
            "chosen_input_ids": chosen_input_ids.to(self.device),
            "chosen_attention_mask": chosen_attention_mask.to(self.device),
            "rejected_input_ids": rejected_input_ids.to(self.device),
            "rejected_attention_mask": rejected_attention_mask.to(self.device),
        }

    def run_inference(self, raw_data, output_path="output.jsonl", batch_size=8):
        """
        Main inference method:
        1. Process raw data
        2. Create DataLoader for batching
        3. Compute s_t values and save to output_path as JSON lines.
        """
        
        fn_kwargs = {
                "processing_class": self.tokenizer,
                "max_prompt_length": self.max_prompt_length,
                "max_completion_length": self.max_completion_length,
                # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                "add_special_tokens": False,
            }
        self.processed_data = raw_data.map(
                self.tokenize_row,
                fn_kwargs=fn_kwargs,
                num_proc=8,
                writer_batch_size=10,
                desc="Tokenizing train dataset",
            )

        dataset = InferenceDataset(self.processed_data)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=self._collate_fn)

        save_jsonl([],output_path,'w')
        for batch in tqdm(loader, desc="Running inference"):
            results = self.compute_s_t(batch)
            save_jsonl(results,output_path,'a')



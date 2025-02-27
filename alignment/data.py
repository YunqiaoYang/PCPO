# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from .configs import DataArguments


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class Processor:
    def __init__(self, tokenizer, max_len, model, dpo_mode=False):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dpo_mode = dpo_mode
        
        if model == "llama3":
            before_query = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            after_query = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            after_response = "<|eot_id|>"
        elif model == "llama3_math":
            before_query = "<|start_header_id|>system<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            after_query = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            after_response = "<|eot_id|>"
        elif model == "llama2":
            before_query = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
            after_query = " [/INST] "
            after_response = " </s>"
        elif model == "Qwen2":
            before_query = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            after_query = "<|im_end|>\n<|im_start|>assistant\n"
            after_response = "<|im_end|>\n"
        elif model == "Qwen2_5":
            before_query = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            after_query = "<|im_end|>\n<|im_start|>assistant\n"
            after_response = "<|im_end|>\n"
        elif model == "Qwen2_5_math":
            before_query = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
            after_query = "<|im_end|>\n<|im_start|>assistant\n"
            after_response = "<|im_end|>\n"
        elif model == "mathstral":
            before_query = ""
            after_query = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            after_response = "<\s>"
        elif model == "deepseek-math":
            before_query = "User: "
            after_query = "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"
            after_response = "<｜end▁of▁sentence｜>"
        else :
            raise ValueError('Not implemented')

        self.before_query = before_query
        self.after_query = after_query
        self.after_response = after_response
        
        self.before_query_id = self.tokenizer.encode(before_query, add_special_tokens=False)
        self.after_query_id = self.tokenizer.encode(after_query, add_special_tokens=False)
        self.after_response_id = self.tokenizer.encode(after_response, add_special_tokens=False)
        

    def _process_tokenize(self, e, prefix=''):


        instruction_id = self.tokenizer.encode(e["instruction"], add_special_tokens=False)
        response_id = self.tokenizer.encode(e["response"], add_special_tokens=False)

        input_ids= self.before_query_id + instruction_id + self.after_query_id + response_id + self.after_response_id
        labels = [-100] * len(self.before_query_id + instruction_id + self.after_query_id) + response_id + self.after_response_id
        
        

        return {f"{prefix}input_ids": input_ids, f"{prefix}labels": labels}

    def process_tokenize(self, e, label_pos, label_neg):
        r = self._process_tokenize(e[label_pos], 'positive_' if self.dpo_mode else '')
        
        if self.dpo_mode:
            r.update(self._process_tokenize(e[label_neg], 'negative_'))
            
        return r
    
    def process(self, e, label_pos, label_neg, **kwargs):
        example={}
        if self.dpo_mode:
            if "prompt" in e:
                prompt_messages = e["prompt"]
                chosen_messages = e[label_pos]
                rejected_messages = e[label_neg]
            else:
                prompt_messages = e[label_pos]["instruction"]
                chosen_messages = e[label_pos]["response"]
                rejected_messages = e[label_neg]["response"]
            # if 'sc_weight' in kwargs:
            #     sc_weight=e[kwargs['sc_weight']]
            #     example.update({"sc_weight":sc_weight})
            example.update({"text_prompt": self.before_query + prompt_messages + self.after_query})
            example.update({"text_chosen": chosen_messages + self.after_response})
            example.update({"text_rejected":rejected_messages + self.after_response})
            example.update({"origin_prompt": prompt_messages})
            example.update({"origin_chosen": chosen_messages})
            example.update({"origin_rejected":rejected_messages})
        else:
            raise ValueError("Other modes not implemented yet")
        return example

    def filter_data(self, e):
        if self.dpo_mode:
            return len(e['positive_input_ids']) <= self.max_len and len(e['negative_input_ids']) <= self.max_len
        
        return len(e['input_ids']) <= self.max_len

def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.get_chat_template()

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task in ["dpo", "orpo"]:
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)


            
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False


def get_datasets(
    data_config: DataArguments | dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'data_config' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer,
        splits=splits,
        configs=configs,
        columns_to_keep=columns_to_keep,
        shuffle=shuffle,
    )
    return raw_datasets


def mix_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split)
            except :
                # If not, check local dataset
                try:
                    dataset = load_from_disk(os.path.join(ds, split))
                except:
                    try:
                        dataset = load_dataset('json', data_files=ds, split=split)
                    except:
                        dataset = load_dataset('parquet', data_files=ds, split=split)

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets

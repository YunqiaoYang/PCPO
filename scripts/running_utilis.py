import yaml
import argparse
import subprocess
import threading
import shlex
import sys
import os
import json
import pandas as pd
import numpy as np
import re
import torch
import ast
from multiprocessing import Process



class ArgYAMLManager:
    def __init__(self, known_args: argparse.Namespace, dynamic_args: dict, skip_kwown_args = False):
        """
        Args:
            known_args (argparse.Namespace): Known arguments from parse_known_args().
            dynamic_args (dict): A dictionary of unknown CLI arguments, e.g.,
                                 {"model.hidden_size": "128", "dataset": ["gsm8k", "math"]}
        """
        # Merge the dynamic_args into known_args so that we have a single place
        # to handle them. We'll store them as a namespace attribute for convenience.
        # If you prefer, you can keep them separate and handle them directly.
        self.known_args = known_args
        self.dynamic_args = dynamic_args
        self.ignore_keys = ['save', 'load', 'update']
        if skip_kwown_args:
            self.ignore_keys = vars(known_args).copy().keys()
        # Convert dynamic_args into an argparse.Namespace, so we can handle it uniformly
        self.dynamic_namespace = argparse.Namespace(**dynamic_args)

    def process_args(self) -> argparse.Namespace:
        """
        Logic:
            1. Convert the known_args + dynamic_args into a hierarchical dictionary (nested keys).
            2. If --update is set, force --load to be True.
            3. If both --save and --load are set, raise an error.
            4. Depending on flags, load existing YAML, update it with dynamic args, etc.
            5. Return the updated namespace.
        """
        # Step 1: Combine the two sets of args into one namespace
        # (the known_args might have load/save flags, etc. The dynamic_namespace has the new param keys.)

        combined_namespace = self.merge_known_and_dynamic()
        
        # Convert to nested dict
        args_dict = self.nested_args_to_dict(combined_namespace)

        # Step 2: If update => load
        if getattr(combined_namespace, 'update', False):
            combined_namespace.load = True
        
        # Step 3: Can't save and load simultaneously
        if getattr(combined_namespace, 'save', False) and getattr(combined_namespace, 'load', False):
            raise ValueError("You can't save YAML and load YAML at once.")

        # Step 4: Handle load / update / save
        existing_args_dict = {}
        if getattr(combined_namespace, 'load', False):
            existing_args_dict = vars(self.load_args_from_yaml(combined_namespace.load_path))
            if getattr(combined_namespace, 'update', False):
                updated_args_ns = self.update_args_with_cli(existing_args_dict, combined_namespace)
        
        if getattr(combined_namespace, 'save', False):
            self.save_args_to_yaml(args_dict, combined_namespace.save_path, self.ignore_keys)
            print("Arguments saved to YAML.")
        elif getattr(combined_namespace, 'update', False):
            if getattr(combined_namespace, 'update_path', False):
                self.save_args_to_yaml(vars(updated_args_ns), combined_namespace.update_path, self.ignore_keys)
                print("Arguments updated to YAML.")
            combined_namespace = updated_args_ns
        elif getattr(combined_namespace, 'load', False):
            combined_namespace = argparse.Namespace(**existing_args_dict)

        return combined_namespace

    def merge_known_and_dynamic(self) -> argparse.Namespace:
        """
        Merge known args namespace and dynamic args dict into a single namespace.
        If a key is present in both, dynamic args can override (depending on your logic).
        """
        merged_dict = vars(self.known_args).copy()
        merged_dict.update(self.dynamic_args)  # dynamic overrides if there's a clash
        return argparse.Namespace(**merged_dict)


    @staticmethod
    def save_args_to_yaml(args_dict: dict, filename: str, ignore_keys = ['save', 'load', 'update']):
        """
        Save the dictionary of arguments to a YAML file.
        Ignores the control fields 'save', 'load', and 'update'.
        
        Args:
            args_dict (dict): The dictionary of arguments to save.
            filename (str): The path to the output YAML file.
        """
        filtered_args = {
            k: v for k, v in args_dict.items()
            if k not in ignore_keys
        }
        with open(filename, 'w', encoding='utf-8') as file:
            yaml.dump(filtered_args, file, allow_unicode=True)

    @staticmethod
    def load_args_from_yaml(filename: str) -> argparse.Namespace:
        """
        Load arguments from a YAML file and return an argparse.Namespace.
        
        Args:
            filename (str): The path to the YAML file.
        
        Returns:
            argparse.Namespace: The loaded arguments as a namespace object.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"YAML file '{filename}' not found.")

        with open(filename, 'r', encoding='utf-8') as file:
            args = yaml.safe_load(file)
        # If the file is empty or could not be parsed, yaml.safe_load might return None
        return argparse.Namespace(**(args if args else {}))

    @staticmethod
    def nested_args_to_dict(args: argparse.Namespace) -> dict:
        """
        Convert a namespace with dot-delimited keys into a hierarchical dictionary.
        For example, a key "model.hidden_size" with value 128 becomes:
        
            {
                "model": {
                    "hidden_size": 128
                }
            }
        
        Args:
            args (argparse.Namespace): The parsed CLI arguments.
        
        Returns:
            dict: A hierarchical dictionary.
        """
        temp_arg_dict = vars(args)
        arg_dict = temp_arg_dict.copy()

        for key in list(arg_dict.keys()):
            parts = key.split('.')
            if len(parts) == 1:
                # No nested keys, skip
                continue

            # Create a nested structure for dot-delimited keys
            current_dict = arg_dict
            for part in parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            current_dict[parts[-1]] = arg_dict[key]

            # Remove the original flat key
            del arg_dict[key]

        return arg_dict

    @staticmethod
    def update_args_with_cli(args_dict: dict, cli_args: argparse.Namespace) -> argparse.Namespace:
        """
        Update args_dict with non-None values from cli_args, supporting dot-delimited nested keys.
        
        Args:
            args_dict (dict): The existing dictionary (usually loaded from a YAML file).
            cli_args (argparse.Namespace): The CLI arguments parsed by argparse.
        
        Returns:
            argparse.Namespace: A namespace with the updated arguments.
        """
        cli_args_dict = vars(cli_args)
        for key, value in cli_args_dict.items():
            if key in ('load', 'save', 'update', 'load_path', 'save_path', 'update_path'): continue
            if value is not None:
                keys = key.split('.')
                current = args_dict
                for subkey in keys[:-1]:
                    if subkey not in current:
                        current[subkey] = {}
                    current = current[subkey]
                current[keys[-1]] = value

        return argparse.Namespace(**args_dict)


def parse_dynamic_args(parser):
    """
    Parse both known and unknown arguments.
    Returns:
        known_args (argparse.Namespace): The known (predefined) arguments.
        dynamic_dict (dict): Key-value pairs derived from unknown arguments.
    """
    # Parse known + unknown arguments
    known_args, unknown_args = parser.parse_known_args()
    # Convert the unknown args to a key-value dictionary
    dynamic_dict = convert_unknown_args_to_dict(unknown_args)
    
    return known_args, dynamic_dict

def parse_single_value(s: str):
    """
    Attempt to parse a single string into a Python object:
    1. Boolean (true/false, case-insensitive)
    2. Integer
    3. Float
    4. Python literal (ast.literal_eval) - can parse valid Python structures like [1,2], {"a": 1}, etc.
    5. String (fallback)
    """
    # 1. Check for bool
    
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    # 2. Try integer
    try:
        return int(s)
    except ValueError:
        pass

    # 3. Try float
    try:
        return float(s)
    except ValueError:
        pass
    # 4. Try ast.literal_eval for other possible structures (e.g. [1,2], {'a': 1}, etc.)
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    # 5. Fallback: treat it as a string
    return s


def parse_value_list(values):
    """
    If there is only one value in the list, parse and return it as a single object.
    If there are multiple values, parse each one and return a list.
    """
    # if len(values) == 1:
    #     return parse_single_value(values[0])
    # else:
        # return [parse_single_value(v) for v in values]
    return parse_single_value(values)

def fix_big_dict(s):
    if s.startswith("{") and s.endswith("}"):
        # Add quotes around keys and values if necessary
        formatted_string = "{"
        content = s[1:-1]  # Extract content inside braces
        parts = content.split(",")  # Split by comma to handle each key-value pair
        formatted_parts = []
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                key = key.strip().strip("'").strip('"')
                value = value.strip()
                formatted_parts.append(f"'{key}': {value}")
        formatted_string += ", ".join(formatted_parts) + "}"
        return formatted_string
    return s
    
def fix_dict(s):
    if s.startswith("{") and s.endswith(":"):
        content = s[1:-1]
        s = "{'"+content+"':"
    return s

def convert_unknown_args_to_dict(unknown_args):
    """
    Converts a list of unknown arguments (e.g. ['--model.hidden_size', '128', '--dataset', 'gsm8k', 'math'])
    into a dict. Example output:

        {
            'model.hidden_size': 128,
            'dataset': ['gsm8k', 'math']
        }

    Logic:
      - Anything that starts with '--' is treated as a key, e.g. '--model.hidden_size' -> 'model.hidden_size'
      - If there is no subsequent token or the next token also starts with '--', treat this key as a boolean flag set to True.
      - Otherwise, gather all subsequent tokens until the next '--' and parse them (via parse_single_value):
          * If there's only one token, we store the parsed single item
          * If multiple tokens, we store a list of parsed items

    Returns:
        dict: key -> parsed_value (str, bool, int, float, list, etc.)
    """
    dynamic_dict = {}
    key = None
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if token.startswith('--'):
            # Start of a new key
            key = token[2:]  # remove the leading '--'
            i += 1

            # If there's no subsequent token or the next token is another '--', treat it as a boolean True
            if i >= len(unknown_args) or unknown_args[i].startswith('--'):
                dynamic_dict[key] = True
            else:
                # Otherwise, gather tokens until the next '--'
                values = ""
                while i < len(unknown_args) and not unknown_args[i].startswith('--'):
                    values=" ".join([values,fix_dict(unknown_args[i])])
                    values = values.strip()
                    i = i + 1
                # Parse and store
                
                dynamic_dict[key] = parse_value_list(values.strip())
                
        else:
            # If the token doesn't start with '--' but there's no key waiting, skip it
            i += 1

    return dynamic_dict


    
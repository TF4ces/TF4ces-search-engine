#!/usr/bin/env python3
# encoding: utf-8

"""
    File Handlers.

    Author : TF4ces
"""


from typing import Dict
import json
from pathlib import Path
import shutil


def read_file(file_path: Path) -> str:
    """
    Read file contents and return as string.
    Args:
        file_path: Absolute path to file.

    Returns:
        content of the file as str.
    """

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    except Exception as e:
        raise Exception(f"Failed to load file {file_path}. Error: {e}")


def load_json(file_path: Path) -> Dict:
    """
    Read file contents and return as Dictionary.
    Args:
        file_path: Absolute path to file.

    Returns:
        content of the file as Dictionary.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)

    except Exception as e:
        raise Exception(f"Failed to load file {file_path}. Error: {e}")


def save_file(file_path: Path, content: str, extension: str = None):
    """
    Saves a file.
    Args:
        file_path: absolute path.
        content: Content to store
        extension: Extension to store as.

    Returns:
        None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'w') as f:
            if extension == 'json':
                json.dump(content, f, indent=2)
            else:
                f.write(content)

    except Exception as e:
        raise Exception(f"Couldn't save file: {file_path} due to: {e}")


def get_all_file_names(dir_path: Path):
    """
    Recursive function to get all files paths present in dir_path.
    Args:
        dir_path: Absolute path to the Directory

    Returns:
        Set of all the file names present in dir_path
    """
    file_names = set()

    for child in dir_path.iterdir():
        if child.is_dir():
            file_names = file_names.union(get_all_file_names(child))
            continue
        if child.stem.startswith('.'):
            continue
        file_names.add(child)

    return file_names


def delete_dir(dir_path: Path):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
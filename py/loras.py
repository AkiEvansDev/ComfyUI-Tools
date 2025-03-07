from nodes import LoraLoader
import re
import os
import folder_paths
from .base import FlexibleOptionalInputType, any_type, is_not_blank, extract_filename

def get_and_strip_loras(prompt, silent=False):
    pattern = "<lora:([^:>]*?)(?::(-?\d*(?:\.\d*)?))?>"
    lora_paths = folder_paths.get_filename_list("loras")

    matches = re.findall(pattern, prompt)

    loras = []
    unfound_loras = []
    skipped_loras = []
    for match in matches:
        tag_path = match[0]

        strength = float(match[1] if len(match) > 1 and len(match[1]) else 1.0)
        if strength == 0:
            skipped_loras.append({"lora": tag_path, "strength": strength})
            continue

        lora_path = get_lora_by_filename(tag_path, lora_paths)
        if lora_path is None:
            unfound_loras.append({"lora": tag_path, "strength": strength})
            continue

        loras.append({"lora": lora_path, "strength": strength})

    return (re.sub(pattern, "", prompt), loras, skipped_loras, unfound_loras)

def get_lora_by_filename(file_path, lora_paths=None):
    lora_paths = lora_paths if lora_paths is not None else folder_paths.get_filename_list("loras")

    if file_path in lora_paths:
        return file_path

    lora_paths_no_ext = [os.path.splitext(x)[0] for x in lora_paths]

    if file_path in lora_paths_no_ext:
        found = lora_paths[lora_paths_no_ext.index(file_path)]
        return found

    file_path_force_no_ext = os.path.splitext(file_path)[0]
    if file_path_force_no_ext in lora_paths_no_ext:
        found = lora_paths[lora_paths_no_ext.index(file_path_force_no_ext)]
        return found

    lora_filenames_only = [os.path.basename(x) for x in lora_paths]
    if file_path in lora_filenames_only:
        found = lora_paths[lora_filenames_only.index(file_path)]
        return found

    file_path_force_filename = os.path.basename(file_path)
    lora_filenames_only = [os.path.basename(x) for x in lora_paths]
    if file_path_force_filename in lora_filenames_only:
        found = lora_paths[lora_filenames_only.index(file_path_force_filename)]
        return found

    lora_filenames_and_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in lora_paths]
    if file_path in lora_filenames_and_no_ext:
        found = lora_paths[lora_filenames_and_no_ext.index(file_path)]
        return found

    file_path_force_filename_and_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    if file_path_force_filename_and_no_ext in lora_filenames_and_no_ext:
        found = lora_paths[lora_filenames_and_no_ext.index(file_path_force_filename_and_no_ext)]
        return found

    for index, lora_path in enumerate(lora_paths):
        if file_path in lora_path:
            found = lora_paths[index]
            return found

    return None

class LorasLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("model", "clip", "names",)
    FUNCTION = "load_loras"
    CATEGORY = "AE.Tools/Lora"

    def load_loras(self, model, clip, **kwargs):
        names = ""
        
        loader = LoraLoader()

        for key, value in kwargs.items():
            key = key.upper()
            if not key.startswith("LORA_"):
                continue

            if "on" in value and "lora" in value and "strength" in value:
                strength_model = value["strength"]
                strength_clip = value["strengthTwo"] if "strengthTwo" in value and value["strengthTwo"] is not None else strength_model
                if value["on"] and (strength_model != 0 or strength_clip != 0):
                    lora = get_lora_by_filename(value["lora"])
                    if lora is not None:
                        model, clip = loader.load_lora(model, clip, lora, strength_model, strength_clip)
                        names += extract_filename(value["lora"]) + ":" + "{:.2f}".format(value["strength"]) + ", "

        if is_not_blank(names):
            names = names.strip(", ")

        return (model, clip, names,)

class CustomLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora": (folder_paths.get_filename_list("loras"), ),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("model", "clip", "name",)
    FUNCTION = "load_lora"
    CATEGORY = "AE.Tools/Lora"

    def load_lora(self, model, clip, lora, strength):
        name = extract_filename(lora) + ":" + "{:.2f}".format(strength)
        model, clip = LoraLoader().load_lora(model, clip, get_lora_by_filename(lora), strength, strength)
        return (model, clip, name,)
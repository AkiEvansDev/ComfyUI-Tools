from nodes import LoraLoader
import re
import os
import folder_paths
from .base import FlexibleOptionalInputType, any_type, is_not_blank, extract_filename, get_path_by_filename

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
                    lora = get_path_by_filename(value["lora"], "loras")
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
                "lora": ([os.path.splitext(path)[0] for path in folder_paths.get_filename_list("loras")], ),
                "strength": ("FLOAT", {"default": 1, "min": -10, "max": 10, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("model", "clip", "name",)
    FUNCTION = "load_lora"
    CATEGORY = "AE.Tools/Lora"

    def load_lora(self, model, clip, lora, strength):
        strength = round(strength, 2)
        name = extract_filename(lora) + ":" + "{:.2f}".format(strength)
        model, clip = LoraLoader().load_lora(model, clip, get_path_by_filename(lora, "loras"), strength, strength)
        return (model, clip, name,)
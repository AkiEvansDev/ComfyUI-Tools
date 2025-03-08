from .checkpoint import CustomCheckpointLoader
from .loras import CustomLoraLoader, get_lora_by_filename
from .base import FlexibleOptionalInputType, any_type, is_not_blank, extract_filename
import comfy.samplers
import folder_paths


class IntList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 99}),
                "list": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, index, list):
        if index is None:
            index = 0

        index = index - 1
        numbers = [int(line.strip()) for line in list.splitlines() if line.strip()]
        return (numbers[index] if 0 <= index < len(numbers) else 0,)

class FloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 99}),
                "list": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, index, list):
        if index is None:
            index = 0

        index = index - 1
        numbers = [round(float(line.strip()), 2) for line in list.splitlines() if line.strip()]
        return (numbers[index] if 0 <= index < len(numbers) else 0,)

class StringList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 99}),
                "list": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, index, list):
        if index is None:
            index = 0

        index = index - 1
        lines = [line.strip() for line in list.splitlines() if line.strip()]
        return (lines[index] if 0 <= index < len(lines) else "",)

class CheckpointList:
    @classmethod
    def INPUT_TYPES(s):
        required = {}
        required["index"] = ("INT", {"default": 1, "min": 1, "max": 99})
        for name in folder_paths.get_filename_list("checkpoints"):
            required[name] = ("BOOLEAN", {"default": True})
        return {"required": required}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("model", "clip", "vae", "name",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, index, **kwargs):
        if index is None:
            index = 0

        index = index - 1
        lines = [key for key, value in kwargs.items() if value == True]

        if 0 <= index < len(lines):
            return CustomCheckpointLoader().load_checkpoint(lines[index])

        return ()

class SamplerList:
    @classmethod
    def INPUT_TYPES(s):
        required = {}
        required["index"] = ("INT", {"default": 1, "min": 1, "max": 99})
        for name in comfy.samplers.KSampler.SAMPLERS:
            required[name] = ("BOOLEAN", {"default": True})
        return {"required": required}

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, index, **kwargs):
        if index is None:
            index = 0

        index = index - 1
        lines = [key for key, value in kwargs.items() if value == True]
        return (lines[index] if 0 <= index < len(lines) else "")

class SchedulerList:
    @classmethod
    def INPUT_TYPES(s):
        required = {}
        required["index"] = ("INT", {"default": 1, "min": 1, "max": 99})
        for name in comfy.samplers.KSampler.SCHEDULERS:
            required[name] = ("BOOLEAN", {"default": True})
        return {"required": required}

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, index, **kwargs):
        if index is None:
            index = 0

        index = index - 1
        lines = [key for key, value in kwargs.items() if value == True]
        return (lines[index] if 0 <= index < len(lines) else "")

class LorasList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "index": ("INT", {"default": 1, "min": 1, "max": 99}),
            },
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("model", "clip", "name",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, model, clip, index, **kwargs):
        if index is None:
            index = 0

        index = index - 1

        values = []
        for key, value in kwargs.items():
            key = key.upper()
            if not key.startswith("LORA_"):
                continue

            if "on" in value and "lora" in value and "strength" in value:
                strength_model = value["strength"]
                strength_clip = value["strengthTwo"] if "strengthTwo" in value and value["strengthTwo"] is not None else strength_model
                if value["on"] and (strength_model != 0 or strength_clip != 0):
                    values.append(value)

        if 0 <= index < len(values):
            return CustomLoraLoader().load_lora(model, clip, get_lora_by_filename(values[index]["lora"]), values[index]["strength"])

        return (model, clip, "None",)
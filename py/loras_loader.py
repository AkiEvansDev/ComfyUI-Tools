from nodes import LoraLoader
from .loras import get_lora_by_filename
from .base import FlexibleOptionalInputType, any_type

class LorasLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": FlexibleOptionalInputType(any_type),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP",)
    RETURN_NAMES = ("model", "clip",)
    FUNCTION = "load_loras"
    CATEGORY = "AE.Tools"

    def load_loras(self, model, clip, **kwargs):
        for key, value in kwargs.items():
            key = key.upper()
            if not key.startswith('LORA_'):
                continue;

            if 'on' in value and 'lora' in value and 'strength' in value:
                strength_model = value['strength']
                strength_clip = value['strengthTwo'] if 'strengthTwo' in value and value['strengthTwo'] is not None else strength_model
                if value['on'] and (strength_model != 0 or strength_clip != 0):
                    lora = get_lora_by_filename(value['lora'])
                    if lora is not None:
                        model, clip = LoraLoader().load_lora(model, clip, lora, strength_model, strength_clip)

        return (model, clip,)

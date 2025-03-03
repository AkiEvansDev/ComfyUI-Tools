from .base import FlexibleOptionalInputType, any_type

def is_none(value):
    if value is not None:
        if isinstance(value, dict) and "model" in value and "clip" in value:
            return not value or all(v is None for v in value.values())
    return value is None

class AnySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("*",)
    FUNCTION = "switch"
    CATEGORY = "AE.Tools/Switch"

    def switch(self, **kwargs):
        any_value = None
        for key, value in kwargs.items():
            if key.startswith("any_") and not is_none(value):
                any_value = value
                break
        return (any_value,)

class IntSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "condition": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = "AE.Tools/Switch"

    def switch(self, a, b, condition):
        return (a if condition == True else b,)

class FloatSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.00, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.05}),
                "b": ("FLOAT", {"default": 0.00, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.05}),
                "condition": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = "AE.Tools/Switch"

    def switch(self, a, b, condition):
        return (a if condition == True else b,)

class StringSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("STRING", {"default": "", "multiline": False}),
                "b": ("STRING", {"default": "", "multiline": False}),
                "condition": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = "AE.Tools/Switch"

    def switch(self, a, b, condition):
        return (a if condition == True else b,)
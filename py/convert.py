from .base import any_type

class IntToFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"forceInput": True}),
            },
        }
                
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Convert"

    def get_value(self, value):
        return (float(value),)

class FloatToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Convert"

    def get_value(self, value):
        return (int(value),)

class ToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (any_type, {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Convert"

    def get_value(self, source=None):
        value = "None"

        if isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)):
            value = str(source)
        elif source is not None:
            try:
                value = str(source)
            except Exception:
                value = "error"

        return (value,)
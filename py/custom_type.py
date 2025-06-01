class Int:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Type"

    def get_value(self, value):
        return (value,)

class Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Type"

    def get_value(self, value):
        value = round(value, 2)
        return (value,)

class String:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Type"

    def get_value(self, value):
        return (value,)

class Text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Type"

    def get_value(self, value):
        return (value,)

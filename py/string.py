class StringLength:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/String"

    def get_value(self, value):
        return (len(value),)

class StringConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("STRING", {"default": '', "multiline": False}),
                "separator": ("STRING", {"default": '', "multiline": False}),
                "b": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/String"

    def get_value(self, a, separator, b):
        return (a + separator + b,)

class StringReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": '', "multiline": False}),
                "find": ("STRING", {"default": '', "multiline": False}),
                "replace": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/String"

    def get_value(self, value, find, replace):
        return (value.replace(find, replace),)

class StringEquals:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("STRING", {"default": '', "multiline": False}),
                "b": ("STRING", {"default": '', "multiline": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/String"

    def get_value(self, a, b):
        return (a == b,)
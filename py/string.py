from .base import is_not_blank, FlexibleOptionalInputType, any_type
from .convert import ToString

class StringLength:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": False}),
            },
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
                "a": ("STRING", {"default": "", "multiline": False}),
                "separator": ("STRING", {"default": "", "multiline": False}),
                "b": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/String"

    def get_value(self, a, separator, b):
        if is_not_blank(a):
            a = a.replace("\\n",  "\n")

        if is_not_blank(separator):
            separator = separator.replace("\\n",  "\n")

        if is_not_blank(b):
            b = b.replace("\\n",  "\n")
            a = a + separator + b 

        return (a,)

class StringReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": False}),
                "find": ("STRING", {"default": "", "multiline": False}),
                "replace": ("STRING", {"default": "", "multiline": False}),
            },
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
                "a": ("STRING", {"default": "", "multiline": False}),
                "b": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/String"

    def get_value(self, a, b):
        return (a == b,)

class ToStringConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {"default": "", "multiline": False}),
                "separator": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/String"

    def get_value(self, template, separator, **kwargs):
        if is_not_blank(template):
            template = template.replace("\\n",  "\n")
        else:
            template = ""

        if is_not_blank(separator):
            separator = separator.replace("\\n",  "\n")
        
        for key, value in kwargs.items():
            if f"{{{key}}}" in template:
                template = template.replace(f"{{{key}}}", ToString.get_value(self, value)[0])
            else:
                template = template + separator + ToString.get_value(self, value)[0]
                
        if is_not_blank(separator):
            template = template.strip(separator)

        return (template,)
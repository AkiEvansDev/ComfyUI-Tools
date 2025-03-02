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
        index = index - 1
        lines = [line.strip() for line in list.splitlines() if line.strip()]
        return (lines[index] if 0 <= index < len(lines) else "",)
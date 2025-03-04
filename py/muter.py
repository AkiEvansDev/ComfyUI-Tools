class GroupsMuter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hidden_text": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("options",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools"

    def get_value(self, hidden_text):
        result = ", ".join(item.split(". ")[-1] for item in hidden_text.split(", "))
        return (result,)
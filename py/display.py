import json
from .base import any_type

class DisplayAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (any_type, {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "display"
    OUTPUT_NODE = True
    CATEGORY = "AE.Tools"

    def display(self, source=None):
        value = "None"

        if isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)):
            value = str(source)
        elif source is not None:
            try:
                value = json.dumps(source)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = "source exists, but could not be serialized."

        return {"ui": {"text": (value,)}}
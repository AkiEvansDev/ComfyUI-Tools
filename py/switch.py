from .base import FlexibleOptionalInputType, any_type

def is_none(value):
    if value is not None:
        if isinstance(value, dict) and 'model' in value and 'clip' in value:
            return not value or all(v is None for v in value.values())
    return value is None

class AnySwitch:
    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ('*',)
    FUNCTION = "switch"
    CATEGORY = "AE.Tools"

    def switch(self, **kwargs):
        any_value = None
        for key, value in kwargs.items():
            if key.startswith('any_') and not is_none(value):
                any_value = value
                break
        return (any_value,)
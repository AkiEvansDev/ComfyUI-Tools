class MathInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "action": (
                    ["a + b", "a - b", "a * b", "a / b"], 
                    {"default": "a + b"}
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Math"

    def get_value(self, a, b, action):
        if action == "a + b":
            a += b
        elif action == "a - b":
            a -= b
        elif action == "a * b":
            a *= b
        elif action == "a / b":
            a = int(a / b if b != 0 else 1)

        return (a,)

class MathFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.05}),
                "b": ("FLOAT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.05}),
                "action": (
                    ["a + b", "a - b", "a * b", "a / b"], 
                    {"default": "a + b"}
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Math"

    def get_value(self, a, b, action):
        a = round(a, 2)
        b = round(b, 2)

        if action == "a + b":
            a += b
        elif action == "a - b":
            a -= b
        elif action == "a * b":
            a *= b
        elif action == "a / b":
            a /= b if b != 0 else 1

        return (a,)

class CompareInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "action": (
                    ["a < b", "a > b", "a == b", "a <= b", "a >= b"], 
                    {"default": "a < b"}
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Math"

    def get_value(self, a, b, action):
        result = False

        if action == "a < b":
            result = a < b
        elif action == "a > b":
            result = a > b
        elif action == "a == b":
            result = a == b
        elif action == "a <= b":
            result = a <= b
        elif action == "a >= b":
            result = a >= b

        return (result,)

class CompareFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.05}),
                "b": ("FLOAT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.05}),
                "action": (
                    ["a < b", "a > b", "a == b", "a <= b", "a >= b"], 
                    {"default": "a < b"}
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Math"

    def get_value(self, a, b, action):
        a = round(a, 2)
        b = round(b, 2)
        result = False

        if action == "a < b":
            result = a < b
        elif action == "a > b":
            result = a > b
        elif action == "a == b":
            result = a == b
        elif action == "a <= b":
            result = a <= b
        elif action == "a >= b":
            result = a >= b

        return (result,)
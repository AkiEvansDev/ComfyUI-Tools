from server import PromptServer
from .server.ae_server import reset_registry

class Range:
    def __init__(self):
        self._current = 0
        self._unique_id = None

    def __del__(self):
        if self._unique_id is not None and self._unique_id in reset_registry:
            reset_registry.pop(self._unique_id, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "start": ("INT", {"default": 1, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "end": ("INT", {"default": 2, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("current",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, current, start, end, unique_id):
        if self._unique_id != unique_id:
            self._unique_id = unique_id

        if unique_id and unique_id not in reset_registry:
            reset_registry[unique_id] = False

        freeze = False
        if unique_id and unique_id in reset_registry and reset_registry[unique_id] == True:
            reset_registry[unique_id] = False
            freeze = True
            self._current = current
        
        current = self._current

        if start > end:
            start, end = end, start

        if current < start or current > end:
            current = start
        else:
            if not freeze:
                current += 1
            if current > end:
                current = start
                
        self._current = current

        PromptServer.instance.send_sync("ae-range-node-feedback", {"node_id": unique_id, "current": current, "start": start, "end": end})

        return (current,)

    @classmethod
    def IS_CHANGED(self, current, start, end, unique_id):
        return float("NaN")

class XYRange:
    def __init__(self):
        self._x = 0
        self._y = 0
        self._unique_id = None

    def __del__(self):
        if self._unique_id is not None and self._unique_id in reset_registry:
            reset_registry.pop(self._unique_id, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "y": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "x_start": ("INT", {"default": 1, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "x_end": ("INT", {"default": 2, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "y_start": ("INT", {"default": 1, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "y_end": ("INT", {"default": 2, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("x", "y",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, x, y, x_start, x_end, y_start, y_end, unique_id):
        if self._unique_id != unique_id:
            self._unique_id = unique_id

        if unique_id and unique_id not in reset_registry:
            reset_registry[unique_id] = False
            
        freeze = False
        if unique_id and unique_id in reset_registry and reset_registry[unique_id] == True:
            reset_registry[unique_id] = False
            freeze = True
            self._x = x
            self._y = y

        x = self._x
        y = self._y

        if x_start > x_end:
            x_start, x_end = x_end, x_start

        if y_start > y_end:
            y_start, y_end = y_end, y_start

        if x < x_start or x > x_end:
            x = x_start

        if y < y_start or y > y_end:
            y = y_start
        else:
            if not freeze:
                y += 1
            if y > y_end:
                y = y_start
                if not freeze:
                    x += 1
                if x > x_end:
                    x = x_start

        self._x = x
        self._y = y
        
        PromptServer.instance.send_sync("ae-xy-range-node-feedback", {"node_id": unique_id, "x": x, "y": y, "x_start": x_start, "x_end": x_end, "y_start": y_start, "y_end": y_end})

        return (x, y,)

    @classmethod
    def IS_CHANGED(selff, x, y, x_start, x_end, y_start, y_end, unique_id):
        return float("NaN")

class RangeList:
    def __init__(self):
        self._current = 0
        self._unique_id = None

    def __del__(self):
        if self._unique_id is not None and self._unique_id in reset_registry:
            reset_registry.pop(self._unique_id, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "list": ("STRING", {"multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("current",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/List"

    def get_value(self, current, list, unique_id):
        if self._unique_id != unique_id:
            self._unique_id = unique_id

        if unique_id and unique_id not in reset_registry:
            reset_registry[unique_id] = False

        freeze = False
        if unique_id and unique_id in reset_registry and reset_registry[unique_id] == True:
            reset_registry[unique_id] = False
            freeze = True
            self._current = current
        
        current = self._current
        numbers = [int(line.strip()) for line in list.splitlines() if line.strip()]
        start = 0
        end = len(numbers) - 1
        index = numbers.index(current) if current in numbers else -1

        if index < start or index > end:
            index = start
        else:
            if not freeze:
                index += 1
            if index > end:
                index = start

        current = numbers[index]
        self._current = current

        PromptServer.instance.send_sync("ae-range-node-feedback", {"node_id": unique_id, "current": current})

        return (current,)

    @classmethod
    def IS_CHANGED(self, current, list, unique_id):
        return float("NaN")
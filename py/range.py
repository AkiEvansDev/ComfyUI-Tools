from server import PromptServer

class Range:
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
        update = False
        
        if start > end:
            start, end = end, start
            update = True

        if update:
            PromptServer.instance.send_sync("ae-range-node-update", {"node_id": unique_id, "start": start, "end": end})

        if current < start or current > end:
            current = start
        else:
            current += 1
            if current > end:
                current = start

        PromptServer.instance.send_sync("ae-range-node-feedback", {"node_id": unique_id, "current": current})

        return (current,)

    @classmethod
    def IS_CHANGED(self, current, start, end, unique_id):
        return float("NaN")

class XYRange:
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
        update = False

        if x_start > x_end:
            x_start, x_end = x_end, x_start
            update = True

        if y_start > y_end:
            y_start, y_end = y_end, y_start
            update = True

        if update:
            PromptServer.instance.send_sync("ae-xyrange-node-update", {"node_id": unique_id, "x_start": x_start, "x_end": x_end, "y_start": y_start, "y_end": y_end})
        
        if x < x_start or x > x_end:
            x = x_start

        if y < y_start or y > y_end:
            y = y_start
        else:
            y += 1
            if y > y_end:
                y = y_start
                x += 1
                if x > x_end:
                    x = x_start

        PromptServer.instance.send_sync("ae-xyrange-node-feedback", {"node_id": unique_id, "x": x, "y": y})

        return (x, y,)

    @classmethod
    def IS_CHANGED(self, x, y, x_start, x_end, y_start, y_end, unique_id):
        return float("NaN")
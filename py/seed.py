from server import PromptServer
from .server.ae_server import reset_registry
import random
from datetime import datetime

initial_random_state = random.getstate()
random.seed(datetime.now().timestamp())
seed_random_state = random.getstate()
random.setstate(initial_random_state)

def new_random_seed():
    global seed_random_state
    random.setstate(seed_random_state)
    seed = random.randint(0, 9999999999999999)
    seed_random_state = random.getstate()

    return seed

class Seed:
    def __init__(self):
        self._seed_value = 0
        self._unique_id = None

    def __del__(self):
        if self._unique_id is not None and self._unique_id in reset_registry:
            reset_registry.pop(self._unique_id, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_value": ("INT", {"default": 0, "min": 0, "max": 9999999999999999}),
                "mode": (
                    ["fixed", "randomize", "increment", "decrement"], 
                    {"default": "fixed"}
                ),
                "even": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Type"

    def get_value(self, seed_value, mode, even, unique_id):
        if self._unique_id != unique_id:
            self._unique_id = unique_id

        if unique_id and unique_id not in reset_registry:
            reset_registry[unique_id] = False

        if unique_id and unique_id in reset_registry and reset_registry[unique_id] == True:
            reset_registry[unique_id] = False
            self._seed_value = seed_value
        else:
            seed_value = self._seed_value

            if mode == "randomize":
                seed_value = new_random_seed()
                if even:
                    seed_value = int(str(seed_value).translate(str.maketrans("13579", "24680")))
            elif mode == "increment":
                seed_value = seed_value + 1 if seed_value < 9999999999999999 else seed_value
            elif mode == "decrement":
                seed_value = seed_value - 1 if seed_value > 0 else seed_value
        
            self._seed_value = seed_value

        PromptServer.instance.send_sync("ae-seed-node-feedback", {"node_id": unique_id, "seed": seed_value})
        
        return (seed_value,)

    @classmethod
    def IS_CHANGED(self, seed_value, mode, even, unique_id):
        if mode == "fixed":
            return 0
        return float("NaN")
from server import PromptServer

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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": 9999999999999999}),
                "mode": (
                    ['fixed', 'randomize', 'increment', 'decrement'], 
                    {"default": 'fixed'}
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
    CATEGORY = "AE.Tools"

    def get_value(self, value, mode, even, unique_id):
        match mode:
            case 'randomize':
                value = new_random_seed()
            case 'increment':
                value = value + 1 if value < 9999999999999999 else value
            case 'decrement':
                value = value - 1 if value > 0 else value
        
        if even:
            value = int(str(value).translate(str.maketrans('13579', '24680')))

        PromptServer.instance.send_sync("ae-seed-node-feedback", {"node_id": unique_id, "seed": value})

        return (value,)
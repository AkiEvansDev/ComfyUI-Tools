from .py.custom_types import *
from .py.list_types import *
from .py.range import *
from .py.seed import *
from .py.switch import *
from .py.display import *
from .py.loras_loader import *

from .py.server.ae_server import *

NODE_CLASS_MAPPINGS = {
    "AE.Int": Int,
    "AE.Float": Float,
    "AE.String": String,
    "AE.Text": Text,
    "AE.IntList": IntList,
    "AE.FloatList": FloatList,
    "AE.StringList": StringList,
    "AE.Range": Range,
    "AE.XYRange": XYRange,
    "AE.SDXLConfig": SDXLConfig,
    "AE.SamplerConfig": SamplerConfig,
    "AE.Seed": Seed,
    "AE.AnySwitch": AnySwitch,
    "AE.DisplayAny": DisplayAny,
    "AE.LorasLoader": LorasLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AE.Int": "Int Value",
    "AE.Float" : "Float Value",
    "AE.String" : "String Value",
    "AE.Text" : "Text Value",
    "AE.IntList": "Int List",
    "AE.FloatList": "Float List",
    "AE.StringList": "String List",
    "AE.Range": "Range",
    "AE.XYRange": "XY Range",
    "AE.SDXLConfig" : "SDXL Config",
    "AE.SamplerConfig" : "Sampler Config",
    "AE.Seed" : "Seed Value",
    "AE.AnySwitch" : "Any Switch",
    "AE.DisplayAny": "Display Any",
    "AE.LorasLoader": "Loras Loader",
}

WEB_DIRECTORY = "js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
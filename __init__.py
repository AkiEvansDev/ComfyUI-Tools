from .py.custom_types import *
from .py.math import *
from .py.string import *
from .py.list_types import *
from .py.convert import *
from .py.range import *
from .py.configs import *
from .py.seed import *
from .py.switch import *
from .py.display import *
from .py.loras_loader import *
from .py.prompt import *

from .py.server.ae_server import *

NODE_CLASS_MAPPINGS = {
    "AE.Int": Int,
    "AE.Float": Float,
    "AE.String": String,
    "AE.Text": Text,
    "AE.MathInt": MathInt,
    "AE.MathFloat": MathFloat,
    "AE.CompareInt": CompareInt,
    "AE.CompareFloat": CompareFloat,
    "AE.StringLength": StringLength,
    "AE.StringConcat": StringConcat,
    "AE.StringReplace": StringReplace,
    "AE.StringEquals": StringEquals,
    "AE.IntList": IntList,
    "AE.FloatList": FloatList,
    "AE.StringList": StringList,
    "AE.IntToFloat": IntToFloat,
    "AE.FloatToInt": FloatToInt,
    "AE.ToString": ToString,
    "AE.Range": Range,
    "AE.XYRange": XYRange,
    "AE.SDXLConfig": SDXLConfig,
    "AE.SamplerConfig": SamplerConfig,
    "AE.Seed": Seed,
    "AE.AnySwitch": AnySwitch,
    "AE.IntSwitch": IntSwitch,
    "AE.FloatSwitch": FloatSwitch,
    "AE.StringSwitch": StringSwitch,
    "AE.DisplayAny": DisplayAny,
    "AE.LorasLoader": LorasLoader,
    "AE.SDXLPrompt": SDXLPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AE.Int": "Int Value",
    "AE.Float" : "Float Value",
    "AE.String" : "String Value",
    "AE.Text" : "Text Value",
    "AE.MathInt": "Math Int",
    "AE.MathFloat": "Math Float",
    "AE.CompareInt": "Compare Int",
    "AE.CompareFloat": "Compare Float",
    "AE.StringLength": "String Length",
    "AE.StringConcat": "String Concat",
    "AE.StringReplace": "String Replace",
    "AE.StringEquals": "String Equals",
    "AE.IntList": "Int List",
    "AE.FloatList": "Float List",
    "AE.StringList": "String List",
    "AE.IntToFloat": "Int To Float",
    "AE.FloatToInt": "Float To Int",
    "AE.ToString": "To String",
    "AE.Range": "Range",
    "AE.XYRange": "XY Range",
    "AE.SDXLConfig" : "SDXL Config",
    "AE.SamplerConfig" : "Sampler Config",
    "AE.Seed" : "Seed Value",
    "AE.AnySwitch" : "Any Switch",
    "AE.IntSwitch": "Int Switch",
    "AE.FloatSwitch": "Float Switch",
    "AE.StringSwitch": "String Switch",
    "AE.DisplayAny": "Display Any",
    "AE.LorasLoader": "Loras Loader",
    "AE.SDXLPrompt": "SDXL Prompt",
}

WEB_DIRECTORY = "js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
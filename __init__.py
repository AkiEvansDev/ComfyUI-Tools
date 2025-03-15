from .py.base import add_folder_path

from .py.custom_type import *
from .py.string import *
from .py.list import *
from .py.seed import *
from .py.config import *
from .py.math import *
from .py.convert import *
from .py.sizes import *
from .py.range import *
from .py.switch import *
from .py.image import *
from .py.muter import *
from .py.display import *
from .py.checkpoint import *
from .py.loras import *
from .py.prompt import *
from .py.controlnet import *
from .py.sampler import *
from .py.upscale import *
from .py.inpaint import *
from .py.outpaint import *

from .py.server.ae_server import *

add_folder_path("inpaint", [".pt", ".pth", ".safetensors", ".patch"])

NODE_CLASS_MAPPINGS = {
    "AE.Int": Int,
    "AE.Float": Float,
    "AE.String": String,
    "AE.Text": Text,

    "AE.StringLength": StringLength,
    "AE.StringConcat": StringConcat,
    "AE.StringReplace": StringReplace,
    "AE.StringEquals": StringEquals,
    "AE.ToStringConcat": ToStringConcat,

    "AE.IntList": IntList,
    "AE.FloatList": FloatList,
    "AE.StringList": StringList,
    "AE.CheckpointList": CheckpointList,
    "AE.SamplerList": SamplerList,
    "AE.SchedulerList": SchedulerList,
    "AE.LorasList": LorasList,
    
    "AE.Seed": Seed,
    
    "AE.SamplerConfig": SamplerConfigNode,
    "AE.SDXLConfig": SDXLConfigNode,
    "AE.ControlNetConfig": ControlNetConfigNode,
    "AE.HiresFixConfig": HiresFixConfigNode,
    "AE.Img2ImgConfig": Img2ImgConfigNode,
    "AE.OutpaintConfig": OutpaintConfigNode,

    "AE.ChangeSamplerConfig": ChangeSamplerConfig,

    "AE.ExtractSamplerConfig": ExtractSamplerConfig,
    "AE.ExtractControlNetConfig": ExtractControlNetConfig,
    "AE.ExtractHiresFixConfig": ExtractHiresFixConfig,
    "AE.ExtractImg2ImgConfig": ExtractImg2ImgConfig,
    "AE.ExtractOutpaintConfig": ExtractOutpaintConfig,

    "AE.MathInt": MathInt,
    "AE.MathFloat": MathFloat,
    "AE.CompareInt": CompareInt,
    "AE.CompareFloat": CompareFloat,
    
    "AE.IntToFloat": IntToFloat,
    "AE.FloatToInt": FloatToInt,
    "AE.ToString": ToString,
    
    "AE.GetLatentSize": GetLatentSize,
    "AE.GetImageSize": GetImageSize,

    "AE.Range": Range,
    "AE.XYRange": XYRange,
    "AE.RangeList": RangeList,

    "AE.AnySwitch": AnySwitch,
    "AE.AnyTypeSwitch": AnyTypeSwitch,
    "AE.IntSwitch": IntSwitch,
    "AE.FloatSwitch": FloatSwitch,
    "AE.StringSwitch": StringSwitch,
    
    "AE.ImageBlank": ImageBlank,
    "AE.ImagePowerNoise": ImagePowerNoise,
    "AE.LoadImageFromPath": LoadImageFromPath,
    "AE.SaveImage": CustomImageSave,
    "AE.ImagePixelate": ImagePixelate,
    "AE.ImageAdjustment": ImageAdjustment,
    "AE.ImageLucySharpen": ImageLucySharpen,
    "AE.ImageStyleFilter": ImageStyleFilter,
    "AE.ImageHighPassFilter": ImageHighPassFilter,
    "AE.ImageLevels": ImageLevels,
    "AE.ImageCannyFilter": ImageCannyFilter,
    "AE.ImageDragonFilter": ImageDragonFilter,
    "AE.ImageBlendMode": ImageBlendMode,
    "AE.ImageBlendMask": ImageBlendMask,
    "AE.GaussianBlurMask": GaussianBlurMask,
    "AE.BRIARemBG": BRIARemBG,
    "AE.BRIARemBGAdvanced": BRIARemBGAdvanced,

    "AE.GroupsMuter": GroupsMuter,

    "AE.DisplayAny": DisplayAny,
    
    "AE.CheckpointLoader": CustomCheckpointLoader,

    "AE.LorasLoader": LorasLoader,
    "AE.LoraLoader": CustomLoraLoader,

    "AE.SDXLPrompt": SDXLPrompt,
    "AE.SDXLPromptWithHires": SDXLPromptWithHires,
    "AE.SDXLRegionalPrompt": SDXLRegionalPrompt,
    "AE.SDXLRegionalPromptWithHires": SDXLRegionalPromptWithHires,
    
    "AE.ControlNetApplyWithConfig": ControlNetApplyWithConfig,

    "AE.KSamplerWithConfig": KSamplerWithConfig,
    "AE.KSamplerHiresFixWithConfig": KSamplerHiresFixWithConfig,
    "AE.KSamplerImg2ImgWithConfig": KSamplerImg2ImgWithConfig,
    "AE.KSamplerInpaintWithConfig": KSamplerInpaintWithConfig,
    "AE.KSamplerOutpaintWithConfigAndImage": KSamplerOutpaintWithConfigAndImage,
    "AE.KSamplerOutpaintWithConfig": KSamplerOutpaintWithConfig,

    "AE.UpscaleLatentBy": UpscaleLatentBy,

    "AE.LoadInpaintModel": LoadInpaintModel,
    "AE.InpaintWithModel": InpaintWithModel,
    "AE.VAEEncodeInpaintConditioning": VAEEncodeInpaintConditioning,

    "AE.OutpaintWithModel": OutpaintWithModel,
    "AE.OutpaintWithModelAndConfig": OutpaintWithModelAndConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AE.Int": "Int Value",
    "AE.Float" : "Float Value",
    "AE.String" : "String Value",
    "AE.Text" : "Text Value",

    "AE.StringLength": "String Length",
    "AE.StringConcat": "String Concat",
    "AE.StringReplace": "String Replace",
    "AE.StringEquals": "String Equals",
    "AE.ToStringConcat": "To String Concat",

    "AE.IntList": "Int List",
    "AE.FloatList": "Float List",
    "AE.StringList": "String List",
    "AE.CheckpointList": "Checkpoint List",
    "AE.SamplerList": "Sampler List",
    "AE.SchedulerList": "Scheduler List",
    "AE.LorasList": "Loras List",
    
    "AE.Seed" : "Seed Value",

    "AE.SamplerConfig": "Sampler Config",
    "AE.SDXLConfig": "SDXL Config",
    "AE.ControlNetConfig": "ControlNet Config",
    "AE.HiresFixConfig": "HiresFix Config",
    "AE.Img2ImgConfig": "Img2Img Config",
    "AE.OutpaintConfig": "Outpaint Config",

    "AE.ChangeSamplerConfig": "Change Sampler Config",

    "AE.ExtractSamplerConfig": "Extract Sampler Config",
    "AE.ExtractControlNetConfig": "Extract ControlNet Config",
    "AE.ExtractHiresFixConfig": "Extract HiresFix Config",
    "AE.ExtractImg2ImgConfig": "Extract Img2Img Config",
    "AE.ExtractOutpaintConfig": "Extract Outpaint Config",

    "AE.MathInt": "Int Math",
    "AE.MathFloat": "Float Math",
    "AE.CompareInt": "Int Compare",
    "AE.CompareFloat": "Float Compare",

    "AE.IntToFloat": "Int To Float",
    "AE.FloatToInt": "Float To Int",
    "AE.ToString": "To String",

    "AE.GetLatentSize": "Latent Size",
    "AE.GetImageSize": "Image Size",

    "AE.Range": "Range",
    "AE.XYRange": "XY Range",
    "AE.RangeList": "Range List",
    
    "AE.AnySwitch" : "Any Switch",
    "AE.AnyTypeSwitch": "AnyType Switch",
    "AE.IntSwitch": "Int Switch",
    "AE.FloatSwitch": "Float Switch",
    "AE.StringSwitch": "String Switch",

    "AE.ImageBlank": "Image Blank",
    "AE.ImagePowerNoise": "Image Power Noise",
    "AE.LoadImageFromPath": "Load Image From Path",
    "AE.SaveImage": "Save Image",
    "AE.ImagePixelate": "Image Pixelate",
    "AE.ImageAdjustment": "Image Adjustment",
    "AE.ImageLucySharpen": "Image Lucy Sharpen",
    "AE.ImageStyleFilter": "Image Style Filter",
    "AE.ImageHighPassFilter": "Image High Pass Filter",
    "AE.ImageLevels": "Image Levels",
    "AE.ImageCannyFilter": "Image Canny Filter",
    "AE.ImageDragonFilter": "Image Dragon Filter",
    "AE.ImageBlendMode": "Image Blend Mode",
    "AE.ImageBlendMask": "Image Blend Mask",
    "AE.GaussianBlurMask": "Gaussian Blur Mask",
    "AE.BRIARemBG": "BRIA Rem BG",
    "AE.BRIARemBGAdvanced": "BRIA Rem BG Advanced",
    
    "AE.GroupsMuter": "Groups Muter",

    "AE.DisplayAny": "Display Any",
    
    "AE.CheckpointLoader": "Load Checkpoint",

    "AE.LorasLoader": "Load Loras",
    "AE.LoraLoader": "Load Lora",

    "AE.SDXLPrompt": "SDXL Prompt",
    "AE.SDXLPromptWithHires": "SDXL Prompt With Hires",
    "AE.SDXLRegionalPrompt": "SDXL Regional Prompt",
    "AE.SDXLRegionalPromptWithHires": "SDXL Regional Prompt With Hires",
    
    "AE.ControlNetApplyWithConfig": "ControlNet Apply With Config",

    "AE.KSamplerWithConfig": "KSampler With Config",
    "AE.KSamplerHiresFixWithConfig": "KSampler HiresFix With Config",
    "AE.KSamplerImg2ImgWithConfig": "KSampler Img2Img With Config",
    "AE.KSamplerInpaintWithConfig": "KSampler Inpaint With Config",
    "AE.KSamplerOutpaintWithConfigAndImage": "KSampler Outpaint With Config & Image",
    "AE.KSamplerOutpaintWithConfig": "KSampler Outpaint With Config",

    "AE.UpscaleLatentBy": "Upscale Latent By Model",
    
    "AE.LoadInpaintModel": "Load Inpaint Model",
    "AE.InpaintWithModel": "Inpaint With Model",
    "AE.VAEEncodeInpaintConditioning": "VAE Encode Inpaint Conditioning",

    "AE.OutpaintWithModel": "Outpaint With Model",
    "AE.OutpaintWithModelAndConfig": "Outpaint With Model & Config",
}

WEB_DIRECTORY = "js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
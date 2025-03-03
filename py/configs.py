from nodes import EmptyLatentImage, MAX_RESOLUTION, ControlNetLoader
import comfy.samplers
import folder_paths
from .base import extract_filename

class SDXLConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dimensions": (
                    [
                        "1536 x 640   (landscape)",
                        "1344 x 768   (landscape)",
                        "1216 x 832   (landscape)",
                        "1152 x 896   (landscape)",
                        "1024 x 1024  (square)",
                        " 896 x 1152  (portrait)",
                        " 832 x 1216  (portrait)",
                        " 768 x 1344  (portrait)",
                        " 640 x 1536  (portrait)",
                    ], 
                    { "default": "1024 x 1024  (square)" }
                ),
                "batch": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            },
        }

    RETURN_TYPES = ("LATENT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("latent", "sampler", "scheduler",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, dimensions, batch, sampler, scheduler):
        result = [x.strip() for x in dimensions.split("x")]
        width = int(result[0])
        height = int(result[1].split(" ")[0])
        latent = EmptyLatentImage().generate(width, height, batch)[0]
        return (latent, sampler, scheduler,)

class SamplerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("steps", "cfg", "denoise",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, steps, cfg, denoise):
        return (steps, cfg, denoise,)

class ControlNetConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("controlnet"), ),
                "strength": ("FLOAT", {"default": 0.25, "min": 0, "max": 2, "step": 0.05}),
                "start": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.05}),
                "end": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET", "FLOAT", "FLOAT", "FLOAT", "STRING",)
    RETURN_NAMES = ("model", "strength", "start", "end", "info",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, model, strength, start, end):
        info = extract_filename(model) + ":" + "{:.2f}".format(strength) + f" [{float(start):.2f}:{float(end):.2f}]"
        control_net, = ControlNetLoader.load_controlnet(self, model)
        return (control_net, strength, start, end, info,)
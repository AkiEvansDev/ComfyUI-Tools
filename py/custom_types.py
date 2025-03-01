from nodes import EmptyLatentImage, MAX_RESOLUTION
import comfy.samplers

class Int:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools"

    def get_value(self, value):
        return (value,)

class Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.00, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools"

    def get_value(self, value):
        return (value,)

class String:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools"

    def get_value(self, string):
        return (string,)

class Text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": '', "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools"

    def get_value(self, string):
        return (string,)


class SDXLConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dimensions": (
                    [
                        '1536 x 640   (landscape)',
                        '1344 x 768   (landscape)',
                        '1216 x 832   (landscape)',
                        '1152 x 896   (landscape)',
                        '1024 x 1024  (square)',
                        ' 896 x 1152  (portrait)',
                        ' 832 x 1216  (portrait)',
                        ' 768 x 1344  (portrait)',
                        ' 640 x 1536  (portrait)',
                    ], 
                    { "default": '1024 x 1024  (square)' }
                ),
                "batch": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            },
        }

    RETURN_TYPES = ("LATENT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("latent", "sampler", "scheduler",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools"

    def get_value(self, dimensions, batch, sampler, scheduler):
        result = [x.strip() for x in dimensions.split('x')]
        width = int(result[0])
        height = int(result[1].split(' ')[0])
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
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("steps", "cfg", "denoise",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools"

    def get_value(self, steps, cfg, denoise):
        return (steps, cfg, denoise,)
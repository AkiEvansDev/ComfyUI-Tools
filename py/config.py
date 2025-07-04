from nodes import EmptyLatentImage, ControlNetLoader
from .inpaint import LoadInpaintModel
import comfy.samplers
import folder_paths
from .base import extract_filename, get_path_by_filename, SamplerConfig, ControlNetConfig, HiresFixConfig, Img2ImgFixConfig, OutpaintConfig, DIMENSIONS, SAMPLER, SCHEDULER, CFG
from .seed import Seed
import os

class ExtractSamplerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("SAMPLER_CONFIG",),
            },
        }

    RETURN_TYPES = ("INT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("seed", "sampler", "scheduler", "steps", "cfg", "denoise",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config/Extract"

    def get_value(self, config):
        return (config.seed, config.sampler, config.scheduler, config.steps, config.cfg, config.denoise,)

class ExtractControlNetConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("CONTROL_NET_CONFIG",),
            },
        }

    RETURN_TYPES = ("CONTROL_NET", "FLOAT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("model", "strength", "start", "end",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config/Extract"

    def get_value(self, config):
        return (config.model, config.strength, config.start, config.end,)

class ExtractHiresFixConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("HIRES_FIX_CONFIG",),
            },
        }

    RETURN_TYPES = ("SAMPLER_CONFIG", "FLOAT",)
    RETURN_NAMES = ("sampler_config", "scale",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config/Extract"

    def get_value(self, config):
        return (config.get_sampler_config(), config.scale,)

class ExtractImg2ImgConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("IMG2IMG_CONFIG",),
            },
        }

    RETURN_TYPES = ("SAMPLER_CONFIG", "CONTROL_NET_CONFIG", "BOOLEAN",)
    RETURN_NAMES = ("sampler_config", "control_net_config", "use_hires_model",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config/Extract"

    def get_value(self, config):
        return (config.get_sampler_config(), config.controlnet, config.use_hires_model,)

class ExtractOutpaintConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("OUTPAINT_CONFIG",),
            },
        }

    RETURN_TYPES = ("SAMPLER_CONFIG", "CONTROL_NET_CONFIG", "INPAINT_MODEL", "INT", "INT", "INT", "INT", "INT", "FLOAT", "BOOLEAN",)
    RETURN_NAMES = ("sampler_config", "control_net_config", "model", "left", "top", "right", "bottom", "feathering", "noise_percentage", "use_hires_model",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config/Extract"

    def get_value(self, config):
        return (config.get_sampler_config(), config.controlnet, config.model, config.left, config.top, config.right, config.bottom, config.feathering, config.noise_percentage, config.use_hires_model,)

class ChangeSamplerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "sampler_config": ("SAMPLER_CONFIG",),
                "hires_fix_config": ("HIRES_FIX_CONFIG",),
                "img2img_config": ("IMG2IMG_CONFIG",),
                "outpaint_config": ("OUTPAINT_CONFIG",),
                "seed_value": ("INT", {"forceInput": True}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True}),
                "steps": ("INT", {"forceInput": True}),
                "cfg": ("FLOAT", {"forceInput": True}),
                "denoise": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("SAMPLER_CONFIG","HIRES_FIX_CONFIG","IMG2IMG_CONFIG","OUTPAINT_CONFIG",)
    RETURN_NAMES = ("sampler_config", "hires_fix_config", "img2img_config", "outpaint_config",)
    FUNCTION = "change_value"
    CATEGORY = "AE.Tools/Config"

    def change_value(self, 
        sampler_config=None, hires_fix_config=None, img2img_config=None, outpaint_config=None, 
        seed_value=None, sampler=None, scheduler=None, steps=None, cfg=None, denoise=None
    ):
        if any(param is not None for param in [seed_value, sampler, scheduler, steps, cfg, denoise]):
            if sampler_config:
                sampler_config = sampler_config.get_copy()
            if hires_fix_config:
                hires_fix_config = hires_fix_config.get_copy()
            if img2img_config:
                img2img_config = img2img_config.get_copy()
            if outpaint_config:
                outpaint_config = outpaint_config.get_copy()

            for config in [sampler_config, hires_fix_config, img2img_config, outpaint_config]:
                if config:
                    if seed_value:
                        config.seed = seed_value
                    if sampler:
                        config.sampler = sampler
                    if scheduler:
                        config.scheduler = scheduler
                    if steps:
                        config.steps = steps
                    if cfg:
                        config.cfg = round(cfg, 2)
                    if denoise:
                        config.denoise = round(denoise, 2)

        return (sampler_config, hires_fix_config, img2img_config, outpaint_config,)

class SamplerConfigNode:
    def __init__(self):
        self.seed = Seed()

    def __del__(self):
        del self.seed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": SAMPLER}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": SCHEDULER}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": CFG, "min": 1, "max": 20, "step": 0.5}),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.05}),
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

    RETURN_TYPES = ("SAMPLER_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, sampler, scheduler, steps, cfg, denoise, seed_value, mode, even, unique_id):
        cfg = round(cfg, 2)
        denoise = round(denoise, 2)

        seed_value, = self.seed.get_value(seed_value, mode, even, unique_id)
        return (SamplerConfig(seed=seed_value, sampler=sampler, scheduler=scheduler, steps=steps, cfg=cfg, denoise=denoise),)

    @classmethod
    def IS_CHANGED(self, sampler, scheduler, steps, cfg, denoise, seed_value, mode, even, unique_id):
        if mode == "fixed":
            return 0
        return float("NaN")

class SDXLConfigNode:
    def __init__(self):
        self.sampler_config = SamplerConfigNode()

    def __del__(self):
        del self.sampler_config

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dimensions": (DIMENSIONS, { "default": "1024 x 1024  (square)" }),
                "seed_value": ("INT", {"default": 0, "min": 0, "max": 9999999999999999}),
                "mode": (
                    ["fixed", "randomize", "increment", "decrement"], 
                    {"default": "fixed"}
                ),
                "even": ("BOOLEAN", {"default": True}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": SAMPLER}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": SCHEDULER}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": CFG, "min": 1, "max": 20, "step": 0.5}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LATENT", "SAMPLER_CONFIG", "STRING",)
    RETURN_NAMES = ("latent", "config", "info",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, dimensions, seed_value, mode, even, sampler, scheduler, steps, cfg, unique_id):
        cfg = round(cfg, 2)
        
        result = [x.strip() for x in dimensions.split("x")]
        width = int(result[0])
        height = int(result[1].split(" ")[0])
        
        latent, = EmptyLatentImage().generate(width, height, 1)
        config, = self.sampler_config.get_value(sampler, scheduler, steps, cfg, 1.0, seed_value, mode, even, unique_id)

        info = f"[Generate]\nDimensions: {width}x{height}\nSeed: {config.seed}\nSampler: {sampler}\nScheduler: {scheduler}\nSteps: {steps}\nCfg: {cfg}"

        return (latent, config, info,)

    @classmethod
    def IS_CHANGED(self, dimensions, seed_value, mode, even, sampler, scheduler, steps, cfg, unique_id):
        if mode == "fixed":
            return 0
        return float("NaN")

class ControlNetConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([os.path.splitext(path)[0] for path in folder_paths.get_filename_list("controlnet")], ),
                "strength": ("FLOAT", {"default": 0.25, "min": 0, "max": 2, "step": 0.05}),
                "start": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.05}),
                "end": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET_CONFIG", "STRING",)
    RETURN_NAMES = ("config", "info",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, model, strength, start, end):
        strength = round(strength, 2)
        start = round(start, 2)
        end = round(end, 2)

        info = extract_filename(model) + f":{strength} [{start}:{end}]"
        control_net, = ControlNetLoader().load_controlnet(get_path_by_filename(model, "controlnet"))
        return (ControlNetConfig(model=control_net, strength=strength, start=start, end=end), info,)

class HiresFixConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scale": ("FLOAT", {"default": 1.5, "min": 1, "max": 4, "step": 0.05}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.05}),
            },
            "optional": {
                "base_config": ("SAMPLER_CONFIG",)
            },
        }

    RETURN_TYPES = ("HIRES_FIX_CONFIG", "STRING",)
    RETURN_NAMES = ("config", "info",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, scale, steps, denoise, base_config=None):
        scale = round(scale, 2)
        denoise = round(denoise, 2)

        config = HiresFixConfig(scale=scale, steps=steps, denoise=denoise)

        if base_config:
            config.seed = base_config.seed
            config.sampler = base_config.sampler
            config.scheduler = base_config.scheduler
            config.cfg = base_config.cfg

        info = f"[HiresFix]\nScale: {scale}\nSteps: {steps}\nDenoise: {denoise}"

        return (config, info,)

class Img2ImgConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_hires_model": ("BOOLEAN", {"default": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "denoise": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "use_control_net": ("BOOLEAN", {"default": True}),
                "controlnet": ([os.path.splitext(path)[0] for path in folder_paths.get_filename_list("controlnet")], ),
                "strength": ("FLOAT", {"default": 0.25, "min": 0, "max": 2, "step": 0.05}),
                "start": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.05}),
                "end": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
            },
            "optional": {
                "base_config": ("SAMPLER_CONFIG",)
            },
        }

    RETURN_TYPES = ("IMG2IMG_CONFIG", "STRING",)
    RETURN_NAMES = ("config", "info",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, use_hires_model, steps, denoise, use_control_net, controlnet, strength, start, end, base_config=None):
        denoise = round(denoise, 2)
        strength = round(strength, 2)
        start = round(start, 2)
        end = round(end, 2)

        config = Img2ImgFixConfig(steps=steps, denoise=denoise, use_hires_model=use_hires_model)
        
        info = "None"
        if use_control_net:
            controlnet_model, info, = ControlNetConfigNode().get_value(controlnet, strength, start, end)
            config.controlnet = controlnet_model

        if base_config:
            config.seed = base_config.seed
            config.sampler = base_config.sampler
            config.scheduler = base_config.scheduler
            config.cfg = base_config.cfg

        info = f"[Img2Img]\nUse Hires Model: {use_hires_model}\nControlNet: {info}\nSteps: {steps}\nDenoise: {denoise}"

        return (config, info,)

class OutpaintConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_hires_model": ("BOOLEAN", {"default": False}),
                "model": ([os.path.splitext(path)[0] for path in folder_paths.get_filename_list("inpaint")],),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100, "step": 1}),
                "denoise": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}),
                "left": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "top": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "right": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "bottom": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "feathering": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "noise_percentage": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_control_net": ("BOOLEAN", {"default": False}),
                "controlnet": ([os.path.splitext(path)[0] for path in folder_paths.get_filename_list("controlnet")], ),
                "strength": ("FLOAT", {"default": 0.25, "min": 0, "max": 2, "step": 0.05}),
                "start": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.05}),
                "end": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
            },
            "optional": {
                "base_config": ("SAMPLER_CONFIG",)
            },
        }

    RETURN_TYPES = ("OUTPAINT_CONFIG", "STRING",)
    RETURN_NAMES = ("config", "info",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, use_hires_model, model, steps, denoise, left, top, right, bottom, feathering, noise_percentage, use_control_net, controlnet, strength, start, end, base_config=None):
        denoise = round(denoise, 2)
        noise_percentage = round(noise_percentage, 2)
        strength = round(strength, 2)
        start = round(start, 2)
        end = round(end, 2)
        
        inpaint, = LoadInpaintModel().load(get_path_by_filename(model, "inpaint"))
        config = OutpaintConfig(model=inpaint, left=left, top=top, right=right, bottom=bottom,
                              feathering=feathering, noise_percentage=noise_percentage, 
                              use_hires_model=use_hires_model, steps=steps, denoise=denoise)

        info = "None"
        if use_control_net:
            controlnet_model, info, = ControlNetConfigNode().get_value(controlnet, strength, start, end)
            config.controlnet = controlnet_model

        if base_config:
            config.seed = base_config.seed
            config.sampler = base_config.sampler
            config.scheduler = base_config.scheduler
            config.cfg = base_config.cfg

        info = (f"[Outpaint]\nGeneral: {extract_filename(model)} [{left}, {top}, {right}, {bottom}] : {feathering} / {noise_percentage}\n"
               f"Use Hires Model: {use_hires_model}\nControlNet: {info}\nSteps: {steps}\nDenoise: {denoise}")

        return (config, info,)
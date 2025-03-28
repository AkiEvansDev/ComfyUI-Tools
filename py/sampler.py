from .controlnet import ControlNetApplyWithConfig
from .upscale import UpscaleLatentBy
from .inpaint import VAEEncodeInpaintConditioning
from .outpaint import OutpaintWithModelAndConfig
from nodes import KSampler, VAEDecode
import comfy.samplers

class KSamplerWithConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "config": ("SAMPLER_CONFIG",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "AE.Tools"

    def sample(self, model, positive, negative, latent, config):
        return KSampler().sample(model, config.seed, config.steps, config.cfg, config.sampler, config.scheduler, positive, negative, latent, config.denoise)

class KSamplerHiresFixWithConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "config": ("HIRES_FIX_CONFIG",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "AE.Tools/Sampler"

    def sample(self, model, positive, negative, latent, config):
        latent, = UpscaleLatentBy().upscale(latent, config.scale)
        return KSampler().sample(model, config.seed, config.steps, config.cfg, config.sampler, config.scheduler, positive, negative, latent, config.denoise)

class KSamplerImg2ImgWithConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "config": ("IMG2IMG_CONFIG",),
            },
            "optional": {
                "model_hires": ("MODEL",),
                "positive_hires": ("CONDITIONING",),
                "negative_hires": ("CONDITIONING",),
                "control_net_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "AE.Tools/Sampler"

    def sample(self, model, positive, negative, latent, vae, config, model_hires=None, positive_hires=None, negative_hires=None, control_net_image=None):
        if config.use_hires_model:
            if model_hires:
                model = model_hires
            if positive_hires:
                positive = positive_hires
            if negative_hires:
                negative = negative_hires
        
        if config.controlnet:
            if control_net_image is None:
                control_net_image, = VAEDecode().decode(vae, latent)
            positive, negative, = ControlNetApplyWithConfig().apply_controlnet(positive, negative, control_net_image, config.controlnet, vae)

        return KSampler().sample(model, config.seed, config.steps, config.cfg, config.sampler, config.scheduler, positive, negative, latent, config.denoise)

class KSamplerInpaintWithConfigAndImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "config": ("IMG2IMG_CONFIG",),
            },
            "optional": {
                "model_hires": ("MODEL",),
                "positive_hires": ("CONDITIONING",),
                "negative_hires": ("CONDITIONING",),
                "control_net_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "AE.Tools/Sampler"

    def sample(self, model, positive, negative, image, mask, vae, config, model_hires=None, positive_hires=None, negative_hires=None, control_net_image=None):
        if config.use_hires_model:
            if model_hires:
                model = model_hires
            if positive_hires:
                positive = positive_hires
            if negative_hires:
                negative = negative_hires
        
        if config.controlnet:
            positive, negative, = ControlNetApplyWithConfig().apply_controlnet(positive, negative, control_net_image if control_net_image is not None else image, config.controlnet, vae)

        positive, negative, latent, = VAEEncodeInpaintConditioning().encode(positive, negative, image, mask, vae)

        return KSampler().sample(model, config.seed, config.steps, config.cfg, config.sampler, config.scheduler, positive, negative, latent, config.denoise)

class KSamplerInpaintWithConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "config": ("IMG2IMG_CONFIG",),
            },
            "optional": {
                "model_hires": ("MODEL",),
                "positive_hires": ("CONDITIONING",),
                "negative_hires": ("CONDITIONING",),
                "control_net_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "AE.Tools/Sampler"

    def sample(self, model, positive, negative, latent, mask, vae, config, model_hires=None, positive_hires=None, negative_hires=None, control_net_image=None):
        image, = VAEDecode().decode(vae, latent)
        return KSamplerInpaintWithConfigAndImage().sample(model, positive, negative, image, mask, vae, config, model_hires, positive_hires, negative_hires, control_net_image)

class KSamplerOutpaintWithConfigAndImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "config": ("OUTPAINT_CONFIG",),
            },
            "optional": {
                "model_hires": ("MODEL",),
                "positive_hires": ("CONDITIONING",),
                "negative_hires": ("CONDITIONING",),
                "control_net_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "AE.Tools/Sampler"

    def sample(self, model, positive, negative, image, vae, config, model_hires=None, positive_hires=None, negative_hires=None, control_net_image=None):
        if config.use_hires_model:
            if model_hires:
                model = model_hires
            if positive_hires:
                positive = positive_hires
            if negative_hires:
                negative = negative_hires
        
        positive, negative, latent, = OutpaintWithModelAndConfig().outpaint(positive, negative, image, vae, config, control_net_image)

        return KSampler().sample(model, config.seed, config.steps, config.cfg, config.sampler, config.scheduler, positive, negative, latent, config.denoise)

class KSamplerOutpaintWithConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "config": ("OUTPAINT_CONFIG",),
            },
            "optional": {
                "model_hires": ("MODEL",),
                "positive_hires": ("CONDITIONING",),
                "negative_hires": ("CONDITIONING",),
                "control_net_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "AE.Tools/Sampler"

    def sample(self, model, positive, negative, latent, vae, config, model_hires=None, positive_hires=None, negative_hires=None, control_net_image=None):
        image, = VAEDecode().decode(vae, latent)
        return KSamplerOutpaintWithConfigAndImage().sample(model, positive, negative, image, vae, config, model_hires, positive_hires, negative_hires, control_net_image)
import numpy as np
import scipy.ndimage
import torch
from .image import ImagePowerNoise, ImageBlendMask
from .inpaint import InpaintWithModel, VAEEncodeInpaintConditioning
from .controlnet import ControlNetApplyWithConfig
from nodes import ImagePadForOutpaint

class OutpaintWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("INPAINT_MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "feathering": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "noise_percentage": ("FLOAT", {"default": 0.1, "min": 0.1, "max": 1.0, "step": 0.01}),
                "vae": ("VAE",),
                "value_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("positive", "negative", "latent",)
    FUNCTION = "outpaint"
    CATEGORY = "AE.Tools/Outpaint"

    def outpaint(self, model, positive, negative, image, left, top, right, bottom, feathering, noise_percentage, vae, value_seed):
        image, mask, = ImagePadForOutpaint().expand_image(image, left, top, right, bottom, feathering)
        image, = InpaintWithModel().inpaint(model, image, mask, value_seed)

        if noise_percentage > 0:
            samples = image.movedim(-1, 1)
            size = samples.shape[3], samples.shape[2]
            noise_image, = ImagePowerNoise().power_noise(size[0], size[1], 0.5, "mix", value_seed)
            image, = ImageBlendMask().image_blend_mask(image, noise_image, mask, noise_percentage)

        if feathering > 0:
            mask, = self.expand_mask(mask, int(feathering / 2), True)

        positive, negative, latent, = VAEEncodeInpaintConditioning().encode(positive, negative, image, mask, vae)

        return (positive, negative, latent,)

    def expand_mask(self, mask, expand, tapered_corners):
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            output = m.numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            output = torch.from_numpy(output)
            out.append(output)
        return (torch.stack(out, dim=0),)

class OutpaintWithModelAndConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "config": ("OUTPAINT_CONFIG",),
            },
            "optional": {
                "control_net_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("positive", "negative", "latent",)
    FUNCTION = "outpaint"
    CATEGORY = "AE.Tools/Outpaint"

    def outpaint(self, positive, negative, image, vae, config, control_net_image=None):
        if config.controlnet:
            positive, negative, = ControlNetApplyWithConfig().apply_controlnet(positive, negative, control_net_image if control_net_image is not None else image, config.controlnet, vae)

        return OutpaintWithModel().outpaint(config.model, positive, negative, image, config.left, config.top, config.right, config.bottom, config.feathering, config.noise_percentage, vae, config.seed)
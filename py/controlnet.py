from nodes import ControlNetApplyAdvanced

class ControlNetApplyWithConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "config": ("CONTROL_NET_CONFIG",),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "AE.Tools"

    def apply_controlnet(self, positive, negative, image, config, vae=None, extra_concat=[]):
        return ControlNetApplyAdvanced().apply_controlnet(positive, negative, config.model, image, config.strength, config.start, config.end, vae, extra_concat)
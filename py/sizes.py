class GetLatentSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "original": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "get_size"
    CATEGORY = "AE.Tools/Size"

    def get_size(self, latent, original):
        lc = latent.copy()
        size = lc["samples"].shape[3], lc["samples"].shape[2]

        if original == False:
            size = size[0] * 8, size[1] * 8

        return (size[0], size[1],)

class GetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "get_size"
    CATEGORY = "AE.Tools/Size"

    def get_size(self, image):
        samples = image.movedim(-1, 1)
        size = samples.shape[3], samples.shape[2]
        return (size[0], size[1],)
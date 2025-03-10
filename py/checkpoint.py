from nodes import CheckpointLoaderSimple
import folder_paths
from .base import extract_filename, get_path_by_filename
import os

class CustomCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([os.path.splitext(path)[0] for path in folder_paths.get_filename_list("checkpoints")], ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("model", "clip", "vae", "name",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "AE.Tools"

    def load_checkpoint(self, model):
        name = extract_filename(model)
        model, clip, vae = CheckpointLoaderSimple().load_checkpoint(get_path_by_filename(model, "checkpoints"))
        return (model, clip, vae, name,)
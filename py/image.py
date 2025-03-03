import torch
import hashlib
from pathlib import Path
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import json
import os
import re
import socket
import time
from .base import is_not_blank
import comfy.model_management
import folder_paths

ALLOWED_EXT = (".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp")

class LoadImageFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("STRING", {"forceInput": True})
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "load_image"
    CATEGORY = "AE.Tools/Image"

    def load_image(self, image):
        image_path = LoadImageFromPath._resolve_path(image)

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

    def _resolve_path(image) -> Path:
        image_path = Path(folder_paths.get_annotated_filepath(image))
        return image_path

    @classmethod
    def IS_CHANGED(s, image):
        image_path = LoadImageFromPath._resolve_path(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if image is None:
            return True

        image_path = LoadImageFromPath._resolve_path(image)
        if not image_path.exists():
            return "Invalid image path: {}".format(image_path)

        return True

class TextTokens:
    def __init__(self):
        self.tokens = {
            "[time]": str(time.time()).replace(".", "_"),
            "[hostname]": socket.gethostname(),
            "[cuda_device]": str(comfy.model_management.get_torch_device()),
            "[cuda_name]": str(comfy.model_management.get_torch_device_name(device=comfy.model_management.get_torch_device())),
        }

        if "." in self.tokens["[time]"]:
            self.tokens["[time]"] = self.tokens["[time]"].split(".")[0]

        try:
            self.tokens["[user]"] = os.getlogin() if os.getlogin() else "null"
        except Exception:
            self.tokens["[user]"] = "null"

    def format_time(self, format_code):
        return time.strftime(format_code, time.localtime(time.time()))

    def parseTokens(self, text):
        tokens = self.tokens.copy()

        tokens["[time]"] = str(time.time())
        if "." in tokens["[time]"]:
            tokens["[time]"] = tokens["[time]"].split(".")[0]

        for token, value in tokens.items():
            if token.startswith("[time("):
                continue
            pattern = re.compile(re.escape(token))
            text = pattern.sub(value, text)

        def replace_custom_time(match):
            format_code = match.group(1)
            return self.format_time(format_code)

        text = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, text)

        return text

class CustomImageSave:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "output_path": ("STRING", {"default": "[time(%Y-%m-%d)]", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "img"}),
                "filename_delimiter": ("STRING", {"default":"_"}),
                "filename_number_padding": ("INT", {"default":4, "min":1, "max":9, "step":1}),
                "filename_number_start": ("BOOLEAN", {"default": True}),
                "extension": (["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp"], {"default": "png"}),
                "dpi": ("INT", {"default": 300, "min": 1, "max": 2400, "step": 1}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "optimize_image": ("BOOLEAN", {"default": False}),
                "lossless_webp": ("BOOLEAN", {"default": False}),
                "embed_workflow": ("BOOLEAN", {"default": False}),
                "save_prompt": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "workflow": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    CATEGORY = "AE.Tools/Image"

    def save_image(
        self,
        images,
        output_path="",
        filename_prefix="img",
        filename_delimiter="_",
        filename_number_padding=4,
        filename_number_start=True,
        extension="png",
        dpi=300,
        quality=100,
        optimize_image=True,
        lossless_webp=False,
        embed_workflow=False,
        save_prompt=False,
        prompt="",
        workflow=None,
        extra_pnginfo=None,
    ):
        tokens = TextTokens()

        delimiter = filename_delimiter
        number_padding = filename_number_padding
        filename_prefix = tokens.parseTokens(filename_prefix)

        if output_path in [None, "", "none", "."]:
            output_path = self.output_dir
        else:
            output_path = tokens.parseTokens(output_path)

        if not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir, output_path)

        if output_path.strip() != "":
            if not os.path.isabs(output_path):
                output_path = os.path.join(folder_paths.output_directory, output_path)
            if not os.path.exists(output_path.strip()):
                print(f"The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.")
                os.makedirs(output_path, exist_ok=True)

        if filename_number_start:
            pattern = f"(\\d+){re.escape(delimiter)}{re.escape(filename_prefix)}"
        else:
            pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d+)"

        existing_counters = [
            int(re.search(pattern, filename).group(1))
            for filename in os.listdir(output_path)
            if re.match(pattern, os.path.basename(filename))
        ]
        existing_counters.sort(reverse=True)

        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        file_extension = "." + extension
        if file_extension not in ALLOWED_EXT:
            print(f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}")
            file_extension = "png"

        results = list()
        output_files = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if extension == "webp":
                img_exif = img.getexif()
                if embed_workflow:
                    workflow_metadata = ""
                    workflow_str = ""
                    if workflow is not None:
                        workflow_str = json.dumps(workflow)
                        img_exif[0x010f] = "Prompt:" + workflow_str
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            workflow_metadata += json.dumps(extra_pnginfo[x])
                    img_exif[0x010e] = "Workflow:" + workflow_metadata
                exif_data = img_exif.tobytes()
            else:
                metadata = PngInfo()
                if embed_workflow:
                    if workflow is not None:
                        metadata.add_text("prompt", json.dumps(workflow))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                exif_data = metadata

            if filename_number_start:
                file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
            else:
                file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"

            if os.path.exists(os.path.join(output_path, file)):
                counter += 1

            try:
                output_file = os.path.abspath(os.path.join(output_path, file))
                if extension in ["jpg", "jpeg"]:
                    img.save(output_file, quality=quality, optimize=optimize_image, dpi=(dpi, dpi))
                elif extension == "webp":
                    img.save(output_file, quality=quality, lossless=lossless_webp, exif=exif_data)
                elif extension == "png":
                    img.save(output_file, pnginfo=exif_data, optimize=optimize_image)
                elif extension == "bmp":
                    img.save(output_file)
                elif extension == "tiff":
                    img.save(output_file, quality=quality, optimize=optimize_image)
                else:
                    img.save(output_file, pnginfo=exif_data, optimize=optimize_image)
                
                if save_prompt and is_not_blank(prompt):
                    with open(output_file + ".txt", "w", encoding="utf-8") as file:
                        file.write(prompt)

                print(f"Image file saved to: {output_file}")
                output_files.append(output_file)

            except OSError as e:
                print(f"Unable to save file to: {output_file}")
                print(e)
            except Exception as e:
                print("Unable to save file due to the to the following error:")
                print(e)

        filtered_paths = []
        
        if filtered_paths:
            for image_path in filtered_paths:
                subfolder = self.get_subfolder_path(image_path, self.output_dir)
                image_data = {
                    "filename": os.path.basename(image_path),
                    "subfolder": subfolder,
                    "type": self.type
                }
                results.append(image_data)

        return {"ui": {"images": []}, "result": ()}

    def get_subfolder_path(self, image_path, output_path):
        output_parts = output_path.strip(os.sep).split(os.sep)
        image_parts = image_path.strip(os.sep).split(os.sep)
        common_parts = os.path.commonprefix([output_parts, image_parts])
        subfolder_parts = image_parts[len(common_parts):]
        subfolder_path = os.sep.join(subfolder_parts[:-1])
        return subfolder_path
from nodes import LoadImage
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps, ImageChops
from PIL.PngImagePlugin import PngInfo
import numpy as np
import json
import os
import random
import re
import socket
import time
import math
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import normalize
from .base import is_not_blank
from .briarmbg import BriaRMBG
import comfy.model_management
import folder_paths

ALLOWED_EXT = (".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp")

class ImageBlank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
                "red": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blank_image"
    CATEGORY = "AE.Tools/Image"

    def blank_image(self, width, height, red, green, blue):
        width = (width // 8) * 8
        height = (height // 8) * 8

        blank = Image.new(mode="RGB", size=(width, height), color=(red, green, blue))

        return (pil2tensor(blank), )

class ImagePowerNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "max": 4096, "min": 64, "step": 1}),
                "height": ("INT", {"default": 1024, "max": 4096, "min": 64, "step": 1}),
                "attenuation": ("FLOAT", {"default": 0.5, "max": 10.0, "min": 0.0, "step": 0.01}),
                "noise_type": (["grey", "white", "pink", "blue", "green", "mix"], {"default": "white"}),
                "value_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "power_noise"
    CATEGORY = "AE.Tools/Image"

    def power_noise(self, width, height, attenuation, noise_type, value_seed):
        return (pil2tensor( self.generate_power_noise(width, height, attenuation, noise_type, value_seed) ), )

    def generate_power_noise(self, width, height, attenuation, noise_type, noise_seed):
        def white_noise(width, height):
            noise = np.random.random((height, width))
            return noise

        def grey_noise(width, height, attenuation):
            noise = np.random.normal(0, attenuation, (height, width))
            return noise

        def blue_noise(width, height, attenuation):
            noise = grey_noise(width, height, attenuation)
            scale = 1.0 / (width * height)
            fy = np.fft.fftfreq(height)[:, np.newaxis] ** 2
            fx = np.fft.fftfreq(width) ** 2
            f = fy + fx
            power = np.sqrt(f)
            power[0, 0] = 1
            noise = np.fft.ifft2(np.fft.fft2(noise) / power)
            noise *= scale / noise.std()
            return np.real(noise)

        def green_noise(width, height, attenuation):
            noise = grey_noise(width, height, attenuation)
            scale = 1.0 / (width * height)
            fy = np.fft.fftfreq(height)[:, np.newaxis] ** 2
            fx = np.fft.fftfreq(width) ** 2
            f = fy + fx
            power = np.sqrt(f)
            power[0, 0] = 1
            noise = np.fft.ifft2(np.fft.fft2(noise) / np.sqrt(power))
            noise *= scale / noise.std()
            return np.real(noise)

        def pink_noise(width, height, attenuation):
            noise = grey_noise(width, height, attenuation)
            scale = 1.0 / (width * height)
            fy = np.fft.fftfreq(height)[:, np.newaxis] ** 2
            fx = np.fft.fftfreq(width) ** 2
            f = fy + fx
            power = np.sqrt(f)
            power[0, 0] = 1
            noise = np.fft.ifft2(np.fft.fft2(noise) * power)
            noise *= scale / noise.std()
            return np.real(noise)

        def blue_noise_mask(width, height, attenuation, seed, num_masks=3):
            masks = []
            for i in range(num_masks):
                mask_seed = seed + i
                np.random.seed(mask_seed)
                mask = blue_noise(width, height, attenuation)
                masks.append(mask)
            return masks

        def blend_noise(width, height, masks, noise_types, attenuations):
            blended_image = Image.new("L", (width, height), color=0)
            fy = np.fft.fftfreq(height)[:, np.newaxis] ** 2
            fx = np.fft.fftfreq(width) ** 2
            f = fy + fx
            i = 0
            for mask, noise_type, attenuation in zip(masks, noise_types, attenuations):
                mask = Image.fromarray((255 * (mask - np.min(mask)) / (np.max(mask) - np.min(mask))).astype(np.uint8).real)
                if noise_type == "white":
                    noise = white_noise(width, height)
                    noise = Image.fromarray((255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))).astype(np.uint8).real)
                elif noise_type == "grey":
                    noise = grey_noise(width, height, attenuation)
                    noise = Image.fromarray((255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))).astype(np.uint8).real)
                elif noise_type == "pink":
                    noise = pink_noise(width, height, attenuation)
                    noise = Image.fromarray((255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))).astype(np.uint8).real)
                elif noise_type == "green":
                    noise = green_noise(width, height, attenuation)
                    noise = Image.fromarray((255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))).astype(np.uint8).real)
                elif noise_type == "blue":
                    noise = blue_noise(width, height, attenuation)
                    noise = Image.fromarray((255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))).astype(np.uint8).real)

                blended_image = Image.composite(blended_image, noise, mask)
                i += 1

            return np.asarray(blended_image)

        def shorten_to_range(value, min_value, max_value):
            range_length = max_value - min_value + 1
            return ((value - min_value) % range_length) + min_value

        if noise_seed > 4294967294:
            noise_seed = shorten_to_range(noise_seed, 0, 4294967293)
            print(f"Seed too large for power noise; rescaled to: {noise_seed}")

        np.random.seed(noise_seed)

        if noise_type == "white":
            noise = white_noise(width, height)
        elif noise_type == "grey":
            noise = grey_noise(width, height, attenuation)
        elif noise_type == "pink":
            noise = pink_noise(width, height, attenuation)
        elif noise_type == "green":
            noise = green_noise(width, height, attenuation)
        elif noise_type == "blue":
            noise = blue_noise(width, height, attenuation)
        elif noise_type == "mix":
            blue_noise_masks = blue_noise_mask(width, height, attenuation, seed=noise_seed, num_masks=3)
            noise_types = ["white", "grey", "pink", "green", "blue"]
            attenuations = [attenuation] * len(noise_types)
            noise = blend_noise(width, height, blue_noise_masks, noise_types, attenuations)

        if noise_type != "mix":
            noise = 255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

        return Image.fromarray(noise.astype(np.uint8).real).convert("RGB")

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
        return LoadImage().load_image(image)

    @classmethod
    def IS_CHANGED(s, image):
        if image is None:
            return 0

        return LoadImage.IS_CHANGED(s, image)

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if image is None:
            return True

        return LoadImage.VALIDATE_INPUTS(s, image)

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
                "name": ("STRING", {"forceInput": True}),
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
        output_path,
        filename_prefix,
        filename_delimiter,
        filename_number_padding,
        filename_number_start,
        extension,
        dpi,
        quality,
        optimize_image,
        lossless_webp,
        embed_workflow,
        save_prompt,
        name=None,
        prompt=None,
        workflow=None,
        extra_pnginfo=None,
    ):
        tokens = TextTokens()

        delimiter = filename_delimiter
        number_padding = filename_number_padding

        if name:
            filename_prefix = f"{name}{filename_delimiter}{filename_prefix}"

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
                
                if save_prompt and prompt and is_not_blank(prompt):
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

def packages(versions=False):
    import sys
    import subprocess
    try:
        result = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.STDOUT)
        lines = result.decode().splitlines()
        return [line if versions else line.split("==")[0] for line in lines]
    except subprocess.CalledProcessError as e:
        print("An error occurred while fetching packages:", e.output.decode())
        return []

def install_package(package, uninstall_first):
    import sys
    import subprocess
    if uninstall_first is None:
        return

    if isinstance(uninstall_first, str):
        uninstall_first = [uninstall_first]

    print(f"Uninstalling {', '.join(uninstall_first)}..")
    subprocess.check_call([sys.executable, "-s", "-m", "pip", "uninstall", *uninstall_first])
    print("Installing package...")
    subprocess.check_call([sys.executable, "-s", "-m", "pip", "-q", "install", package])

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImagePixelate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "pixelation_size": ("INT", {"default": 128, "min": 16, "max": 480, "step": 1}),
                "num_colors": ("INT", {"default": 32, "min": 2, "max": 256, "step": 1}),
                "init_mode": (["k-means++", "random", "none"], {"default": "random"}),
                "dither": ("BOOLEAN", {"default": False}),
                "dither_mode": (["floyd", "ordered"], {"default": "floyd"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "image_pixelate"
    CATEGORY = "AE.Tools/Image/Filter"

    def image_pixelate(self, images, pixelation_size, num_colors, init_mode, dither, dither_mode):
        if "scikit-learn" not in packages():
            install_package("scikit-learn")

        tensors = []
        for image in images:
            tensors.append(self.pixel_art(image, pixelation_size, num_colors, init_mode, dither, dither_mode))

        return (torch.cat(tensors, dim=0),)

    def pixel_art(
        self, 
        image, 
        min_size, 
        num_colors, 
        init_mode,
        dither, 
        dither_mode
    ):
        from sklearn.cluster import KMeans

        def flatten_colors(image, num_colors, init_mode="random", max_iter=100, random_state=42):
            np_image = np.array(image)
            pixels = np_image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=num_colors, init=init_mode, max_iter=max_iter, tol=1e-3, random_state=random_state, n_init="auto")
            labels = kmeans.fit_predict(pixels)
            colors = kmeans.cluster_centers_.astype(np.uint8)
            flattened_pixels = colors[labels]
            flattened_image = flattened_pixels.reshape(np_image.shape)
            return Image.fromarray(flattened_image)

        def dither_image(image, mode, nc):

            def clamp(value, min_value=0, max_value=255):
                return max(min(value, max_value), min_value)

            def get_new_val(old_val, nc):
                return np.round(old_val * (nc - 1)) / (nc - 1)

            def fs_dither(img, nc):
                arr = np.array(img, dtype=float) / 255
                new_width, new_height = img.size

                for ir in range(new_height):
                    for ic in range(new_width):
                        old_val = arr[ir, ic].copy()
                        new_val = get_new_val(old_val, nc)
                        arr[ir, ic] = new_val
                        err = old_val - new_val

                        if ic < new_width - 1:
                            arr[ir, ic + 1] += err * 7/16
                        if ir < new_height - 1:
                            if ic > 0:
                                arr[ir + 1, ic - 1] += err * 3/16
                            arr[ir + 1, ic] += err * 5/16
                            if ic < new_width - 1:
                                arr[ir + 1, ic + 1] += err / 16

                carr = np.array(arr * 255, dtype=np.uint8)
                return Image.fromarray(carr)

            def ordered_dither(img, nc):
                width, height = img.size
                dithered_image = Image.new("RGB", (width, height))
                num_colors = min(2 ** int(np.log2(nc)), 16)

                for y in range(height):
                    for x in range(width):
                        old_pixel = img.getpixel((x, y))
                        new_pixel = tuple(int(c * num_colors / 256) * (256 // num_colors) for c in old_pixel)
                        dithered_image.putpixel((x, y), new_pixel)

                        if x < width - 1:
                            neighboring_pixel = img.getpixel((x + 1, y))
                            neighboring_pixel = tuple(int(c * num_colors / 256) * (256 // num_colors) for c in neighboring_pixel)
                            neighboring_error = tuple(neighboring - new for neighboring, new in zip(neighboring_pixel, new_pixel))
                            neighboring_pixel = tuple(int(clamp(pixel + error * 7 / 16)) for pixel, error in zip(neighboring_pixel, neighboring_error))
                            img.putpixel((x + 1, y), neighboring_pixel)

                        if x < width - 1 and y < height - 1:
                            neighboring_pixel = img.getpixel((x + 1, y + 1))
                            neighboring_pixel = tuple(int(c * num_colors / 256) * (256 // num_colors) for c in neighboring_pixel)
                            neighboring_error = tuple(neighboring - new for neighboring, new in zip(neighboring_pixel, new_pixel))
                            neighboring_pixel = tuple(int(clamp(pixel + error * 1 / 16)) for pixel, error in zip(neighboring_pixel, neighboring_error))
                            img.putpixel((x + 1, y + 1), neighboring_pixel)

                        if y < height - 1:
                            neighboring_pixel = img.getpixel((x, y + 1))
                            neighboring_pixel = tuple(int(c * num_colors / 256) * (256 // num_colors) for c in neighboring_pixel)
                            neighboring_error = tuple(neighboring - new for neighboring, new in zip(neighboring_pixel, new_pixel))
                            neighboring_pixel = tuple(int(clamp(pixel + error * 5 / 16)) for pixel, error in zip(neighboring_pixel, neighboring_error))
                            img.putpixel((x, y + 1), neighboring_pixel)

                        if x > 0 and y < height - 1:
                            neighboring_pixel = img.getpixel((x - 1, y + 1))
                            neighboring_pixel = tuple(int(c * num_colors / 256) * (256 // num_colors) for c in neighboring_pixel)
                            neighboring_error = tuple(neighboring - new for neighboring, new in zip(neighboring_pixel, new_pixel))
                            neighboring_pixel = tuple(int(clamp(pixel + error * 3 / 16)) for pixel, error in zip(neighboring_pixel, neighboring_error))
                            img.putpixel((x - 1, y + 1), neighboring_pixel)

                return dithered_image

            if mode == "floyd":
                return fs_dither(image, nc)
            elif mode == "ordered":
                return ordered_dither(image, nc)
            else:
                print(f"Inavlid dithering mode `{mode}` selected.")

            return image

        image = tensor2pil(image)
        width, height = image.size

        if max(width, height) > min_size:
            if width > height:
                new_width = min_size
                new_height = int(height * (min_size / width))
            else:
                new_height = min_size
                new_width = int(width * (min_size / height))
            image = image.resize((new_width, int(new_height)), Image.NEAREST)

        if init_mode != "none":
            image = flatten_colors(image, num_colors, init_mode)

        if dither:
            image = dither_image(image, dither_mode, num_colors)
        
        image = image.resize((width, height), Image.NEAREST)

        return pil2tensor(image)

class ImageAdjustment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1, "min": -1, "max": 2, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1, "min": 0, "max": 5, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 1, "min": -5, "max": 5, "step": 0.01}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "gaussian_blur": ("FLOAT", {"default": 0, "min": 0, "max": 1024, "step": 0.1}),
                "edge_enhance": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "detail_enhance": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "image_filters"
    CATEGORY = "AE.Tools/Image/Filter"

    def image_filters(self, images, brightness, contrast, saturation, sharpness, blur, gaussian_blur, edge_enhance, detail_enhance):
        tensors = []
        for image in images:
            pil_image = None

            if brightness > 0.0 or brightness < 0.0:
                image = np.clip(image + brightness, 0.0, 1.0)

            if contrast > 1.0 or contrast < 1.0:
                image = np.clip(image * contrast, 0.0, 1.0)

            if saturation > 1.0 or saturation < 1.0:
                pil_image = tensor2pil(image)
                pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

            if sharpness > 1.0 or sharpness < 1.0:
                pil_image = pil_image if pil_image else tensor2pil(image)
                pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

            if blur > 0:
                pil_image = pil_image if pil_image else tensor2pil(image)
                for _ in range(blur):
                    pil_image = pil_image.filter(ImageFilter.BLUR)

            if gaussian_blur > 0.0:
                pil_image = pil_image if pil_image else tensor2pil(image)
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))

            if edge_enhance > 0.0:
                pil_image = pil_image if pil_image else tensor2pil(image)
                edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                blend_mask = Image.new(mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
                pil_image = Image.composite(edge_enhanced_img, pil_image, blend_mask)
                del blend_mask, edge_enhanced_img

            if detail_enhance:
                pil_image = pil_image if pil_image else tensor2pil(image)
                pil_image = pil_image.filter(ImageFilter.DETAIL)

            tensors.append(pil2tensor(pil_image) if pil_image else image.unsqueeze(0))

        return (torch.cat(tensors, dim=0), )

class ImageLucySharpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "iterations": ("INT", {"default": 12, "min": 1, "max": 16, "step": 1}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sharpen"
    CATEGORY = "AE.Tools/Image/Filter"

    def sharpen(self, images, iterations, kernel_size):
        tensors = []
        for image in images:
            tensors.append(self.lucy_sharpen(image, iterations, kernel_size))

        return (torch.cat(tensors, dim=0),)

    def lucy_sharpen(self, image, iterations=10, kernel_size=3):
        from scipy.signal import convolve2d

        image = tensor2pil(image)

        image_array = np.array(image, dtype=np.float32) / 255.0
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        sharpened_channels = []

        padded_image_array = np.pad(image_array, ((kernel_size, kernel_size), (kernel_size, kernel_size), (0, 0)), mode="edge")

        for channel in range(3):
            channel_array = padded_image_array[:, :, channel]

            for _ in range(iterations):
                blurred_channel = convolve2d(channel_array, kernel, mode="same")
                ratio = channel_array / (blurred_channel + 1e-6)
                channel_array *= convolve2d(ratio, kernel, mode="same")

            sharpened_channels.append(channel_array)

        cropped_sharpened_image_array = np.stack(sharpened_channels, axis=-1)[kernel_size:-kernel_size, kernel_size:-kernel_size, :]
        sharpened_image_array = np.clip(cropped_sharpened_image_array * 255.0, 0, 255).astype(np.uint8)
        sharpened_image = Image.fromarray(sharpened_image_array)

        return pil2tensor(sharpened_image)

class ImageStyleFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "style": ([
                    "1977",
                    "aden",
                    "brannan",
                    "brooklyn",
                    "clarendon",
                    "earlybird",
                    "fairy tale",
                    "gingham",
                    "hudson",
                    "inkwell",
                    "kelvin",
                    "lark",
                    "lofi",
                    "maven",
                    "mayfair",
                    "moon",
                    "nashville",
                    "perpetua",
                    "reyes",
                    "rise",
                    "slumber",
                    "stinson",
                    "toaster",
                    "valencia",
                    "walden",
                    "willow",
                    "xpro2"
                ], {"default": "fairy tale"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "image_style_filter"
    CATEGORY = "AE.Tools/Image/Filter"

    def image_style_filter(self, images, style):
        if "pilgram" not in packages():
            install_package("pilgram")

        import pilgram

        tensors = []
        for image in images:
            image = tensor2pil(image)
            if style == "1977":
                tensors.append(pil2tensor(pilgram._1977(image)))
            elif style == "aden":
                tensors.append(pil2tensor(pilgram.aden(image)))
            elif style == "brannan":
                tensors.append(pil2tensor(pilgram.brannan(image)))
            elif style == "brooklyn":
                tensors.append(pil2tensor(pilgram.brooklyn(image)))
            elif style == "clarendon":
                tensors.append(pil2tensor(pilgram.clarendon(image)))
            elif style == "earlybird":
                tensors.append(pil2tensor(pilgram.earlybird(image)))
            elif style == "fairy tale":
                tensors.append(pil2tensor(self.sparkle(image)))
            elif style == "gingham":
                tensors.append(pil2tensor(pilgram.gingham(image)))
            elif style == "hudson":
                tensors.append(pil2tensor(pilgram.hudson(image)))
            elif style == "inkwell":
                tensors.append(pil2tensor(pilgram.inkwell(image)))
            elif style == "kelvin":
                tensors.append(pil2tensor(pilgram.kelvin(image)))
            elif style == "lark":
                tensors.append(pil2tensor(pilgram.lark(image)))
            elif style == "lofi":
                tensors.append(pil2tensor(pilgram.lofi(image)))
            elif style == "maven":
                tensors.append(pil2tensor(pilgram.maven(image)))
            elif style == "mayfair":
                tensors.append(pil2tensor(pilgram.mayfair(image)))
            elif style == "moon":
                tensors.append(pil2tensor(pilgram.moon(image)))
            elif style == "nashville":
                tensors.append(pil2tensor(pilgram.nashville(image)))
            elif style == "perpetua":
                tensors.append(pil2tensor(pilgram.perpetua(image)))
            elif style == "reyes":
                tensors.append(pil2tensor(pilgram.reyes(image)))
            elif style == "rise":
                tensors.append(pil2tensor(pilgram.rise(image)))
            elif style == "slumber":
                tensors.append(pil2tensor(pilgram.slumber(image)))
            elif style == "stinson":
                tensors.append(pil2tensor(pilgram.stinson(image)))
            elif style == "toaster":
                tensors.append(pil2tensor(pilgram.toaster(image)))
            elif style == "valencia":
                tensors.append(pil2tensor(pilgram.valencia(image)))
            elif style == "walden":
                tensors.append(pil2tensor(pilgram.walden(image)))
            elif style == "willow":
                tensors.append(pil2tensor(pilgram.willow(image)))
            elif style == "xpro2":
                tensors.append(pil2tensor(pilgram.xpro2(image)))

        return (torch.cat(tensors, dim=0), )

    def sparkle(self, image):
        if "pilgram" not in packages():
            install_package("pilgram")

        import pilgram

        image = image.convert("RGBA")
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(1.25)
        saturation_enhancer = ImageEnhance.Color(image)
        image = saturation_enhancer.enhance(1.5)

        bloom = image.filter(ImageFilter.GaussianBlur(radius=20))
        bloom = ImageEnhance.Brightness(bloom).enhance(1.2)
        bloom.putalpha(128)
        bloom = bloom.convert(image.mode)
        image = Image.alpha_composite(image, bloom)

        width, height = image.size

        particles = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(particles)
        for _ in range(5000):
            x = random.randint(0, width)
            y = random.randint(0, height)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            draw.point((x, y), fill=(r, g, b, 255))
        particles = particles.filter(ImageFilter.GaussianBlur(radius=1))
        particles.putalpha(128)

        particles2 = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(particles2)
        for _ in range(5000):
            x = random.randint(0, width)
            y = random.randint(0, height)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            draw.point((x, y), fill=(r, g, b, 255))
        particles2 = particles2.filter(ImageFilter.GaussianBlur(radius=1))
        particles2.putalpha(128)

        image = pilgram.css.blending.color_dodge(image, particles)
        image = pilgram.css.blending.lighten(image, particles2)

        return image

class ImageHighPassFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "radius": ("INT", {"default": 10, "min": 1, "max": 500, "step": 1}),
                "strength": ("FLOAT", {"default": 1.5, "min": 0, "max": 255, "step": 0.1}),
                "color_output": ("BOOLEAN", {"default": True}),
                "neutral_background": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "high_pass"
    CATEGORY = "AE.Tools/Image/Filter"

    def high_pass(self, images, radius, strength, color_output, neutral_background):
        tensors = []
        for image in images:
            tensors.append(self.apply_hpf(image, radius, strength, color_output, neutral_background))
        
        return (torch.cat(tensors, dim=0), )

    def apply_hpf(self, image, radius, strength, color_output, neutral_background):
        image = tensor2pil(image)
        
        image_arr = np.array(image).astype("float")
        blurred_arr = np.array(image.filter(ImageFilter.GaussianBlur(radius=radius))).astype("float")
        hpf_arr = image_arr - blurred_arr
        hpf_arr = np.clip(hpf_arr * strength, 0, 255).astype("uint8")

        if color_output:
            high_pass = Image.fromarray(hpf_arr, mode="RGB")
        else:
            grayscale_arr = np.mean(hpf_arr, axis=2).astype("uint8")
            high_pass = Image.fromarray(grayscale_arr, mode="L")

        if neutral_background:
            neutral_color = (128, 128, 128) if high_pass.mode == "RGB" else 128
            neutral_bg = Image.new(high_pass.mode, high_pass.size, neutral_color)
            high_pass = ImageChops.screen(neutral_bg, high_pass)

        return pil2tensor(high_pass.convert("RGB"))

class ImageLevels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "black_level": ("FLOAT", {"default": 0, "min": 0, "max": 255, "step": 0.1}),
                "mid_level": ("FLOAT", {"default": 127.5, "min": 0, "max": 255, "step": 0.1}),
                "white_level": ("FLOAT", {"default": 255, "min": 0, "max": 255, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_image_levels"
    CATEGORY = "AE.Tools/Image/Filter"

    def apply_image_levels(self, images, black_level, mid_level, white_level):
        tensors = []
        for image in images:
            tensors.append(self.adjust(image, black_level, mid_level, white_level))

        return (torch.cat(tensors, dim=0), )

    def adjust(self, image, min_level, mid_level, max_level):
        image = tensor2pil(image)

        image_arr = np.array(image).astype(np.float32)
        image_arr[image_arr < min_level] = min_level
        image_arr = (image_arr - min_level) * \
            (255 / (max_level - min_level))
        image_arr = np.clip(image_arr, 0, 255)

        if mid_level <= min_level:  
            gamma = 1.0
        else:
            gamma = math.log(0.5) / math.log((mid_level - min_level) / (max_level - min_level))

        image_arr = np.power(image_arr / 255, gamma) * 255
        image_arr = image_arr.astype(np.uint8)

        return pil2tensor(Image.fromarray(image_arr))

class ImageCannyFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["normal", "laplacian"], {"default": "normal"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "canny_filter"
    CATEGORY = "AE.Tools/Image/Filter"

    def canny_filter(self, images, mode):
        tensors = []
        for image in images:
            if mode == "laplacian":
                image = tensor2pil(image)
                image = image.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                                                -1, -1, -1, -1), 1, 0))
                tensors.append(torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0))
            else:
                tensors.append(self.canny_detector(255. * image.cpu().numpy().squeeze()))

        return (torch.cat(tensors, dim=0),)

    def canny_detector(self, image):
        import cv2

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 1.4)
        gx = cv2.Sobel(np.float32(image), cv2.CV_64F, 1, 0, 3)
        gy = cv2.Sobel(np.float32(image), cv2.CV_64F, 0, 1, 3)

        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        height, width = image.shape

        for i_x in range(width):
            for i_y in range(height):
                grad_ang = ang[i_y, i_x]
                grad_ang = abs(grad_ang-180) if abs(grad_ang) > 180 else abs(grad_ang)

                neighb_1_x, neighb_1_y = -1, -1
                neighb_2_x, neighb_2_y = -1, -1

                if grad_ang <= 22.5:
                    neighb_1_x, neighb_1_y = i_x-1, i_y
                    neighb_2_x, neighb_2_y = i_x + 1, i_y

                elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                    neighb_1_x, neighb_1_y = i_x-1, i_y-1
                    neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
                elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                    neighb_1_x, neighb_1_y = i_x, i_y-1
                    neighb_2_x, neighb_2_y = i_x, i_y + 1
                elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                    neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                    neighb_2_x, neighb_2_y = i_x + 1, i_y-1
                elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                    neighb_1_x, neighb_1_y = i_x-1, i_y
                    neighb_2_x, neighb_2_y = i_x + 1, i_y
                if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                    if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                        mag[i_y, i_x] = 0
                        continue

                if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                    if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                        mag[i_y, i_x] = 0

        return pil2tensor(Image.fromarray(mag).convert("RGB"))

class ImageDragonFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "saturation": ("FLOAT", {"default": 1, "min": 0, "max": 16, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1, "min": 0, "max": 16, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1, "min": 0, "max": 16, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 1, "min": 0, "max": 6, "step": 0.01}),
                "highpass_samples": ("INT", {"default": 1, "min": 0, "max": 6, "step": 1}),
                "highpass_strength": ("FLOAT", {"default": 1, "min": 0, "max": 3, "step": 0.01}),
                "highpass_radius": ("FLOAT", {"default": 6, "min": 0, "max": 255, "step": 0.01}),
                "colorize": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_dragan_filter"
    CATEGORY = "AE.Tools/Image/Filter"

    def apply_dragan_filter(self, images, saturation, contrast, sharpness, brightness, highpass_samples, highpass_strength, highpass_radius, colorize):
        tensors = []
        for image in images:
            tensors.append(self.dragan_filter(image, saturation, contrast, sharpness, brightness, highpass_samples, highpass_strength, highpass_radius, colorize))

        return (torch.cat(tensors, dim=0), )

    def dragan_filter(self, image, saturation, contrast, sharpness, brightness, highpass_samples, highpass_strength, highpass_radius, colorize):
        if "pilgram" not in packages():
            install_package("pilgram")

        import pilgram

        image = tensor2pil(image)

        alpha = None
        if image.mode == "RGBA":
            alpha = image.getchannel("A")

        grayscale_image = image if image.mode == "L" else image.convert("L")

        contrast_enhancer = ImageEnhance.Contrast(grayscale_image)
        contrast_image = contrast_enhancer.enhance(contrast)

        saturation_enhancer = ImageEnhance.Color(contrast_image) if image.mode != "L" else None
        saturation_image = contrast_image if saturation_enhancer is None else saturation_enhancer.enhance(saturation)

        sharpness_enhancer = ImageEnhance.Sharpness(saturation_image)
        sharpness_image = sharpness_enhancer.enhance(sharpness)

        brightness_enhancer = ImageEnhance.Brightness(sharpness_image)
        brightness_image = brightness_enhancer.enhance(brightness)

        blurred_image = brightness_image.filter(ImageFilter.GaussianBlur(radius=-highpass_radius))
        highpass_filter = ImageChops.subtract(image, blurred_image.convert("RGB"))
        blank_image = Image.new("RGB", image.size, (127, 127, 127))
        highpass_image = ImageChops.screen(blank_image, highpass_filter.resize(image.size))

        if not colorize:
            highpass_image = highpass_image.convert("L").convert("RGB")

        highpassed_image = pilgram.css.blending.overlay(brightness_image.convert("RGB"), highpass_image)
        for _ in range((highpass_samples if highpass_samples > 0 else 1)):
            highpassed_image = pilgram.css.blending.overlay(highpassed_image, highpass_image)

        final_image = ImageChops.blend(brightness_image.convert("RGB"), highpassed_image, highpass_strength)

        if colorize:
            final_image = pilgram.css.blending.color(final_image, image)

        if alpha:
            final_image.putalpha(alpha)

        return pil2tensor(final_image)

class ImageBlendMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mode": ([
                    "add",
                    "color",
                    "color_burn",
                    "color_dodge",
                    "darken",
                    "difference",
                    "exclusion",
                    "hard_light",
                    "hue",
                    "lighten",
                    "multiply",
                    "overlay",
                    "screen",
                    "soft_light"
                ], {"default": "color"}),
                "blend_percentage": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_blend"
    CATEGORY = "AE.Tools/Image"

    def image_blend(self, image_a, image_b, mode, blend_percentage):
        if "pilgram" not in packages():
            install_package("pilgram")

        import pilgram

        image_a = tensor2pil(image_a)
        image_b = tensor2pil(image_b)

        if mode == "color":
            out_image = pilgram.css.blending.color(image_a, image_b)
        elif mode == "color_burn":
            out_image = pilgram.css.blending.color_burn(image_a, image_b)
        elif mode == "color_dodge":
            out_image = pilgram.css.blending.color_dodge(image_a, image_b)
        elif mode == "darken":
            out_image = pilgram.css.blending.darken(image_a, image_b)
        elif mode == "difference":
            out_image = pilgram.css.blending.difference(image_a, image_b)
        elif mode == "exclusion":
            out_image = pilgram.css.blending.exclusion(image_a, image_b)
        elif mode == "hard_light":
            out_image = pilgram.css.blending.hard_light(image_a, image_b)
        elif mode == "hue":
            out_image = pilgram.css.blending.hue(image_a, image_b)
        elif mode == "lighten":
            out_image = pilgram.css.blending.lighten(image_a, image_b)
        elif mode == "multiply":
            out_image = pilgram.css.blending.multiply(image_a, image_b)
        elif mode == "add":
            out_image = pilgram.css.blending.normal(image_a, image_b)
        elif mode == "overlay":
            out_image = pilgram.css.blending.overlay(image_a, image_b)
        elif mode == "screen":
            out_image = pilgram.css.blending.screen(image_a, image_b)
        elif mode == "soft_light":
            out_image = pilgram.css.blending.soft_light(image_a, image_b)
        else:
            out_image = image_a

        out_image = out_image.convert("RGB")

        blend_mask = Image.new(mode="L", size=image_a.size, color=(round(blend_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        out_image = Image.composite(image_a, out_image, blend_mask)

        return (pil2tensor(out_image), )

class ImageBlendMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mask": ("MASK",),
                "blend_percentage": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_blend_mask"
    CATEGORY = "AE.Tools/Image"

    def image_blend_mask(self, image_a, image_b, mask, blend_percentage):
        image_a = tensor2pil(image_a)
        image_b = tensor2pil(image_b)

        image_mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        image_mask = ImageOps.invert(tensor2pil(image_mask).convert("L"))

        masked_img = Image.composite(image_a, image_b, image_mask.resize(image_a.size))

        blend_mask = Image.new(mode="L", size=image_a.size, color=(round(blend_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        img_result = Image.composite(image_a, masked_img, blend_mask)

        del image_a, image_b, blend_mask, masked_img, mask, image_mask

        return (pil2tensor(img_result), )

class GaussianBlurMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", ),
                "kernel_size": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "sigma": ("FLOAT", {"default": 10, "min": 0.1, "max": 100, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "blur_mask"
    CATEGORY = "AE.Tools/Image"

    def blur_mask(self, mask, kernel_size, sigma):
        mask = self.make_3d_mask(mask)
        mask = torch.unsqueeze(mask, dim=-1)
        mask = self.tensor_gaussian_blur_mask(mask, kernel_size, sigma)
        mask = torch.squeeze(mask, dim=-1)
        return (mask, )

    def make_3d_mask(self, mask):
        if len(mask.shape) == 4:
            return mask.squeeze(0)
        elif len(mask.shape) == 2:
            return mask.unsqueeze(0)

        return mask

    def tensor_gaussian_blur_mask(self, mask, kernel_size, sigma):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if mask.ndim == 2:
            mask = mask[None, ..., None]
        elif mask.ndim == 3:
            mask = mask[..., None]

        if mask.ndim != 4:
            raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions!")
        if mask.shape[-1] != 1:
            raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels!")

        if kernel_size <= 0:
            return mask

        kernel_size = kernel_size*2+1

        shortest = min(mask.shape[1], mask.shape[2])
        if shortest <= kernel_size:
            kernel_size = int(shortest/2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 3:
                return mask

        prev_device = mask.device
        device = comfy.model_management.get_torch_device()
        mask.to(device)

        mask = mask[:, None, ..., 0]
        blurred_mask = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(mask)
        blurred_mask = blurred_mask[:, 0, ..., None]

        blurred_mask.to(prev_device)

        return blurred_mask

class BRIARemBG:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "bria_rembg_v1.4.pth")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "remove_background"
    CATEGORY = "AE.Tools/Image"
  
    def remove_background(self, images):
        processed_images = []
        processed_masks = []

        if not self.model:
            self.model = BriaRMBG()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

        for image in images:
            orig_image = tensor2pil(image)
            w,h = orig_image.size

            image = self.resize_image(orig_image)
            im_np = np.array(image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
            im_tensor = torch.unsqueeze(im_tensor,0)
            im_tensor = torch.divide(im_tensor,255.0)
            im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                im_tensor=im_tensor.cuda()

            result = self.model(im_tensor)
            result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)    
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
            new_im.paste(orig_image, mask=pil_im)

            new_im_tensor = pil2tensor(new_im)  
            pil_im_tensor = pil2tensor(pil_im)  

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        return (torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0), )

    def resize_image(self, image):
        image = image.convert('RGB')
        model_input_size = (1024, 1024)
        image = image.resize(model_input_size, Image.BILINEAR)
        return image

class BRIARemBGAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "iterations": ("INT", {"default": 14, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "remove_background"
    CATEGORY = "AE.Tools/Image"
  
    def remove_background(self, images, iterations):
        processed_images, = ImageLucySharpen().sharpen(images, iterations, 3)
        processed_images, processed_masks, = BRIARemBG().remove_background(processed_images)

        return self.join_image_with_alpha(images, processed_masks)

    def join_image_with_alpha(self, image: torch.Tensor, alpha: torch.Tensor):
        batch_size = min(len(image), len(alpha))
        out_images = []

        alpha = self.resize_mask(alpha, image.shape[1:])
        for i in range(batch_size):
           out_images.append(torch.cat((image[i][:,:,:3], alpha[i].unsqueeze(2)), dim=2))

        return (torch.stack(out_images),)

    def resize_mask(self, mask, shape):
        return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[0], shape[1]), mode="bilinear").squeeze(1)


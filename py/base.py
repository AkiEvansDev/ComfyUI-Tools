import os
from symtable import SymbolTable
import folder_paths

class SamplerConfig:
    def __init__(self, seed=0, sampler="", scheduler="", steps=30, cfg=5.0, denoise=1.0):
        self.seed = seed
        self.sampler = sampler
        self.scheduler = scheduler
        self.steps = steps
        self.cfg = cfg
        self.denoise = denoise

    def get_sampler_config(self):
        return SamplerConfig(self.seed, self.sampler, self.scheduler, self.steps, self.cfg, self.denoise)

    def get_copy(self):
        return self.get_sampler_config()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        if value >= 0:
            self._seed = value
        else:
            raise ValueError("Seed must be non-negative!")

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        self._sampler = value

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = value

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, value):
        if value > 0:
            self._steps = value
        else:
            raise ValueError("Steps must be greater than 0!")

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, value):
        if value > 0.0:
            self._cfg = value
        else:
            raise ValueError("CFG must be greater than 0!")

    @property
    def denoise(self):
        return self._denoise

    @denoise.setter
    def denoise(self, value):
        if 0.0 <= value <= 1.0:
            self._denoise = value
        else:
            raise ValueError("Denoise must be between 0.0 and 1.0!")

    def __str__(self):
        return (f"(seed={self.seed}, sampler={self.sampler}, scheduler={self.scheduler}" 
                f", steps={self.steps}, cfg={self.cfg}, denoise={self.denoise})")

class ControlNetConfig:
    def __init__(self, model=None, strength=0.25, start=0.0, end=0.5):
        self.model = model
        self.strength = strength
        self.start = start
        self.end = end

    def get_copy(self):
        return ControlNetConfig(self.model, self.strength, self.start, self.end)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, value):
        if 0.0 <= value <= 1.0:
            self._strength = value
        else:
            raise ValueError("Strength must be between 0.0 and 1.0!")

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        if 0.0 <= value <= 1.0:
            self._start = value
        else:
            raise ValueError("Start must be between 0.0 and 1.0!")

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if 0.0 <= value <= 1.0:
            self._end = value
        else:
            raise ValueError("End must be between 0.0 and 1.0!")

    def __str__(self):
        return (f"(model={self.model}, strength={self.strength}, start={self.start}, end={self.end})")

class SamplerConfigWithControlNet(SamplerConfig):
    def __init__(self, controlnet=None, seed=0, sampler="", scheduler="", steps=30, cfg=5.0, denoise=1.0):
        super().__init__(seed, sampler, scheduler, steps, cfg, denoise)
        self.controlnet = controlnet

    @property
    def controlnet(self):
        return self._controlnet

    @controlnet.setter
    def controlnet(self, value):
        self._controlnet = value

    def __str__(self):
        return (f"{self.controlnet} + {super().__str__()}")

class HiresFixConfig(SamplerConfig):
    def __init__(self, seed=0, sampler="", scheduler="", steps=15, cfg=5.0, denoise=0.4, scale=1.5):
        super().__init__(seed, sampler, scheduler, steps, cfg, denoise)
        self.scale = scale

    def get_copy(self):
        return HiresFixConfig(self.seed, self.sampler, self.scheduler, self.steps, self.cfg, self.denoise, self.scale)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value > 0:
            self._scale = value
        else:
            raise ValueError("Scale must be greater than 0!")

    def __str__(self):
        return (f"(scale={self.scale}) + {super().__str__()}")

class Img2ImgFixConfig(SamplerConfigWithControlNet):
    def __init__(self, seed=0, sampler="", scheduler="", steps=20, cfg=7.0, denoise=1.0, use_hires_model=True):
        super().__init__(None, seed, sampler, scheduler, steps, cfg, denoise)
        self.use_hires_model = use_hires_model

    def get_copy(self):
        copy = Img2ImgFixConfig(self.seed, self.sampler, self.scheduler, self.steps, self.cfg, self.denoise, self.use_hires_model)
        if self.controlnet:
            copy.controlnet = self.controlnet.get_copy()
        return copy

    @property
    def use_hires_model(self):
        return self._use_hires_model

    @use_hires_model.setter
    def use_hires_model(self, value):
        if isinstance(value, bool):
            self._use_hires_model = value
        else:
            raise ValueError("use_hires_model must be a boolean!")

    def __str__(self):
        return (f"(use_hires_model={self.use_hires_model}) + {super().__str__()}")

class OutpaintConfig(SamplerConfigWithControlNet):
    def __init__(self, seed=0, sampler="", scheduler="", steps=30, cfg=5.0, denoise=0.6, model=None,
                 left=0, top=0, right=0, bottom=0, 
                 feathering=64, noise_percentage=0.1, use_hires_model=True):
        super().__init__(None, seed, sampler, scheduler, steps, cfg, denoise)
        self.model = model
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.feathering = feathering
        self.noise_percentage = noise_percentage
        self.use_hires_model = use_hires_model

    def get_copy(self):
        copy = OutpaintConfig(self.seed, self.sampler, self.scheduler, self.steps, self.cfg, self.denoise,
                              self.model, self.left, self.top, self.right, self.bottom, 
                              self.feathering, self.noise_percentage, self.use_hires_model)
        if self.controlnet:
            copy.controlnet = self.controlnet.get_copy()
        return copy

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        if value >= 0:
            self._left = value
        else:
            raise ValueError("Left must be non-negative!")

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, value):
        if value >= 0:
            self._top = value
        else:
            raise ValueError("Top must be non-negative!")

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        if value >= 0:
            self._right = value
        else:
            raise ValueError("Right must be non-negative!")

    @property
    def bottom(self):
        return self._bottom

    @bottom.setter
    def bottom(self, value):
        if value >= 0:
            self._bottom = value
        else:
            raise ValueError("Bottom must be non-negative!")

    @property
    def feathering(self):
        return self._feathering

    @feathering.setter
    def feathering(self, value):
        if value >= 0:
            self._feathering = value
        else:
            raise ValueError("Feathering must be non-negative!")

    @property
    def noise_percentage(self):
        return self._noise_percentage

    @noise_percentage.setter
    def noise_percentage(self, value):
        if 0.0 <= value <= 1.0:
            self._noise_percentage = value
        else:
            raise ValueError("Noise percentage must be between 0.0 and 1.0!")

    @property
    def use_hires_model(self):
        return self._use_hires_model

    @use_hires_model.setter
    def use_hires_model(self, value):
        if isinstance(value, bool):
            self._use_hires_model = value
        else:
            raise ValueError("use_hires_model must be a boolean!")

    def __str__(self):
        return (f"(model={self.model}, left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom}, "
                f"feathering={self.feathering}, noise_percentage={self.noise_percentage}, use_hires_model={self.use_hires_model}) + {super().__str__()}")

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class FlexibleOptionalInputType(dict):
    def __init__(self, type):
        self.type = type

    def __getitem__(self, key):
        return (self.type, )

    def __contains__(self, key):
        return True

any_type = AnyType("*")

def path_exists(path):
    if path is not None:
        return os.path.exists(path)
    return False

def is_none(value):
    if isinstance(value, str):
        return is_blank(value)

    if value is not None:
        if isinstance(value, dict) and "model" in value and "clip" in value:
            return not value or all(v is None for v in value.values())

    return value is None

def is_blank(s):
    return not (s and s.strip())

def is_not_blank(s):
    return bool(s and s.strip())

def clean_string(s):
    if is_not_blank(s):
        s = " ".join(s.strip().split())
    return s

def is_not_blank_replace(s, find, replace):
    if is_not_blank(s) and is_not_blank(find) and is_not_blank(replace):
        s = s.replace(find, replace)
    return s

def extract_filename(filepath):
    parts = filepath.split("\\")
    filename_with_ext = parts[-1]
    return filename_with_ext.split(".")[0]

def filter_non_empty_strings(strings):
    return [s for s in strings if s.strip()]

def add_folder_path(folder_name: str, extensions_to_register: list):
    path = os.path.join(folder_paths.models_dir, folder_name)
    folders, extensions = folder_paths.folder_names_and_paths.get(folder_name, ([], set()))
    
    if path not in folders:
        folders.append(path)
    if isinstance(extensions, set):
        extensions.update(extensions_to_register)
    elif isinstance(extensions, list):
        extensions.extend(extensions_to_register)
    else:
        e = f"Failed to register models/inpaint folder. Found existing value: {extensions}"
        raise Exception(e)

    folder_paths.folder_names_and_paths[folder_name] = (folders, extensions)
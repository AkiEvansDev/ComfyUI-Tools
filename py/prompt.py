from nodes import CLIPTextEncode, ConditioningCombine, ConditioningConcat
from .base import is_not_blank, clean_string, is_not_blank_replace

def get_prompt(prompt, posture, background, style, negative, negative_style):
    text = ""

    if is_not_blank(prompt):
        text += f"[Prompt]\n{prompt}\n\n"

    if is_not_blank(posture):
        text += f"[Posture]\n{posture}\n\n"
        
    if is_not_blank(background):
        text += f"[Background]\n{background}\n\n"
        
    if is_not_blank(style):
        text += f"[Style]\n{style}\n\n"
        
    if is_not_blank(negative):
        text += f"[Negative]\n{negative}\n\n"

    if is_not_blank(negative_style):
        text += f"[Negative Style]\n{negative_style}\n\n"

    return text

class SDXLPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "find_1": ("STRING", {"default": "", "multiline": False}),
                "find_2": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "posture": ("STRING", {"default": "", "multiline": True}),
                "background": ("STRING", {"default": "", "multiline": True}),
                "style": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "negative_style": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "replace_1": ("STRING", {"forceInput": True}),
                "replace_2": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING",)
    RETURN_NAMES = ("positive", "negative", "prompt",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, clip, find_1, find_2, prompt, posture, background, style, negative, negative_style, replace_1="", replace_2=""):
        replace_1 = clean_string(replace_1)
        replace_2 = clean_string(replace_2)
        find_1 = clean_string(find_1)
        find_2 = clean_string(find_2)

        prompt = clean_string(prompt)
        posture = clean_string(posture)
        background = clean_string(background)
        style = clean_string(style)
        negative = clean_string(negative)
        negative_style = clean_string(negative_style)

        prompt = is_not_blank_replace(prompt, find_1, replace_1)
        posture = is_not_blank_replace(posture, find_1, replace_1)
        background = is_not_blank_replace(background, find_1, replace_1)

        prompt = is_not_blank_replace(prompt, find_2, replace_2)
        posture = is_not_blank_replace(posture, find_2, replace_2)
        background = is_not_blank_replace(background, find_2, replace_2)

        text_encode = CLIPTextEncode()
        conditioning_concat = ConditioningConcat()
        conditioning_combine = ConditioningCombine()

        positive_con, = text_encode.encode(clip, prompt)
        negative_con, = text_encode.encode(clip, negative_style)

        if is_not_blank(posture):
            posture_con, = text_encode.encode(clip, posture)
            positive_con, = conditioning_concat.concat(positive_con, posture_con)

        if is_not_blank(background) and is_not_blank(style):
            background_con, = text_encode.encode(clip, background)
            style_con, = text_encode.encode(clip, style)
            background_style_con, = conditioning_concat.concat(background_con, style_con)
            positive_con, = conditioning_combine.combine(positive_con, background_style_con)
        elif is_not_blank(background):
            background_con, = text_encode.encode(clip, background)
            positive_con, = conditioning_combine.combine(positive_con, background_con)
        elif is_not_blank(style):
            style_con, = text_encode.encode(clip, style)
            positive_con, = conditioning_combine.combine(positive_con, style_con)

        if is_not_blank(negative):
            negative_con_base, = text_encode.encode(clip, negative)
            negative_con, = conditioning_concat.concat(negative_con_base, negative_con)
        
        prompt_text = get_prompt(prompt, posture, background, style, negative, negative_style)

        return (positive_con, negative_con, prompt_text,)

class SDXLPromptWithHires:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_hires": ("CLIP",),
                "find_1": ("STRING", {"default": "", "multiline": False}),
                "find_2": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "posture": ("STRING", {"default": "", "multiline": True}),
                "background": ("STRING", {"default": "", "multiline": True}),
                "style": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "negative_style": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "replace_1": ("STRING", {"forceInput": True}),
                "replace_2": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "STRING",)
    RETURN_NAMES = ("positive", "negative", "positive_hires", "negative_hires", "prompt",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools/Config"

    def get_value(self, clip, clip_hires, find_1, find_2, prompt, posture, background, style, negative, negative_style, replace_1="", replace_2=""):
        replace_1 = clean_string(replace_1)
        replace_2 = clean_string(replace_2)
        find_1 = clean_string(find_1)
        find_2 = clean_string(find_2)

        prompt = clean_string(prompt)
        posture = clean_string(posture)
        background = clean_string(background)
        style = clean_string(style)
        negative = clean_string(negative)
        negative_style = clean_string(negative_style)

        prompt = is_not_blank_replace(prompt, find_1, replace_1)
        posture = is_not_blank_replace(posture, find_1, replace_1)
        background = is_not_blank_replace(background, find_1, replace_1)

        prompt = is_not_blank_replace(prompt, find_2, replace_2)
        posture = is_not_blank_replace(posture, find_2, replace_2)
        background = is_not_blank_replace(background, find_2, replace_2)

        text_encode = CLIPTextEncode()
        conditioning_concat = ConditioningConcat()
        conditioning_combine = ConditioningCombine()

        positive_con, = text_encode.encode(clip, prompt)
        positive_con_hires, = text_encode.encode(clip_hires, prompt)
        negative_con, = text_encode.encode(clip, negative_style)
        negative_con_hires, = text_encode.encode(clip_hires, negative_style)

        if is_not_blank(posture):
            posture_con, = text_encode.encode(clip, posture)
            posture_con_hires, = text_encode.encode(clip_hires, posture)
            positive_con, = conditioning_concat.concat(positive_con, posture_con)
            positive_con_hires, = conditioning_concat.concat(positive_con_hires, posture_con_hires)

        if is_not_blank(background) and is_not_blank(style):
            background_con, = text_encode.encode(clip, background)
            background_con_hires, = text_encode.encode(clip_hires, background)
            style_con, = text_encode.encode(clip, style)
            style_con_hires, = text_encode.encode(clip_hires, style)
            background_style_con, = conditioning_concat.concat(background_con, style_con)
            background_style_con_hires, = conditioning_concat.concat(background_con_hires, style_con_hires)
            positive_con, = conditioning_combine.combine(positive_con, background_style_con)
            positive_con_hires, = conditioning_combine.combine(positive_con_hires, background_style_con_hires)
        elif is_not_blank(background):
            background_con, = text_encode.encode(clip, background)
            background_con_hires, = text_encode.encode(clip_hires, background)
            positive_con, = conditioning_combine.combine(positive_con, background_con)
            positive_con_hires, = conditioning_combine.combine(positive_con_hires, background_con_hires)
        elif is_not_blank(style):
            style_con, = text_encode.encode(clip, style)
            style_con_hires, = text_encode.encode(clip_hires, style)
            positive_con, = conditioning_combine.combine(positive_con, style_con)
            positive_con_hires, = conditioning_combine.combine(positive_con_hires, style_con_hires)

        if is_not_blank(negative):
            negative_con_base, = text_encode.encode(clip, negative)
            negative_con_base_hires, = text_encode.encode(clip_hires, negative)
            negative_con, = conditioning_concat.concat(negative_con_base, negative_con)
            negative_con_hires, = conditioning_concat.concat(negative_con_base_hires, negative_con_hires)
        
        prompt_text = get_prompt(prompt, posture, background, style, negative, negative_style)

        return (positive_con, negative_con, positive_con_hires, negative_con_hires, prompt_text,)
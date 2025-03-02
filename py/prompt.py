from nodes import CLIPTextEncode, ConditioningCombine, ConditioningConcat
from .base import is_not_blank, clean_string, is_not_blank_replace, filter_non_empty_strings

class SDXLPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "find_1": ("STRING", {"default": '', "multiline": False}),
                "find_2": ("STRING", {"default": '', "multiline": False}),
                "character": ("STRING", {"default": '', "multiline": True}),
                "posture": ("STRING", {"default": '', "multiline": True}),
                "background": ("STRING", {"default": '', "multiline": True}),
                "style": ("STRING", {"default": '', "multiline": True}),
                "negative": ("STRING", {"default": '', "multiline": True}),
                "negative_style": ("STRING", {"default": '', "multiline": True}),
            },
            "optional": {
                "replace_1": ("STRING", {"forceInput": True}),
                "replace_2": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING",)
    RETURN_NAMES = ("positive", "negative", "prompt",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools"

    def get_value(self, clip, find_1, find_2, character, posture, background, style, negative, negative_style, replace_1="", replace_2=""):
        replace_1 = clean_string(replace_1)
        replace_2 = clean_string(replace_2)
        find_1 = clean_string(find_1)
        find_2 = clean_string(find_2)

        character = clean_string(character)
        posture = clean_string(posture)
        background = clean_string(background)
        style = clean_string(style)
        negative = clean_string(negative)
        negative_style = clean_string(negative_style)

        character = is_not_blank_replace(character, find_1, replace_1)
        posture = is_not_blank_replace(posture, find_1, replace_1)
        background = is_not_blank_replace(background, find_1, replace_1)

        character = is_not_blank_replace(character, find_2, replace_2)
        posture = is_not_blank_replace(posture, find_2, replace_2)
        background = is_not_blank_replace(background, find_2, replace_2)

        positive_con, = CLIPTextEncode.encode(self, clip, character)
        negative_con, = CLIPTextEncode.encode(self, clip, negative_style)

        if is_not_blank(posture):
            posture_con, = CLIPTextEncode.encode(self, clip, posture)
            positive_con, = ConditioningConcat.concat(self, positive_con, posture_con)

        if is_not_blank(background) and is_not_blank(style):
            background_con, = CLIPTextEncode.encode(self, clip, background)
            style_con, = CLIPTextEncode.encode(self, clip, style)
            background_style_con, = ConditioningConcat.concat(self, background_con, style_con)
            positive_con, = ConditioningCombine.combine(self, positive_con, background_style_con)
        elif is_not_blank(background):
            background_con, = CLIPTextEncode.encode(self, clip, background)
            positive_con, = ConditioningCombine.combine(self, positive_con, background_con)
        elif is_not_blank(style):
            style_con, = CLIPTextEncode.encode(self, clip, style)
            positive_con, = ConditioningCombine.combine(self, positive_con, style_con)

        if is_not_blank(negative):
            negative_con_base, = CLIPTextEncode.encode(self, clip, negative)
            negative_con, = ConditioningConcat.concat(self, negative_con_base, negative_con)
        
        prompt = "\r\n\r\n".join(filter_non_empty_strings([character, posture, background, style, negative, negative_style]))

        return (positive_con, negative_con, prompt,)

class SDXLPromptWithHires:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_hires": ("CLIP",),
                "clip": ("CLIP",),
                "find_1": ("STRING", {"default": '', "multiline": False}),
                "find_2": ("STRING", {"default": '', "multiline": False}),
                "character": ("STRING", {"default": '', "multiline": True}),
                "posture": ("STRING", {"default": '', "multiline": True}),
                "background": ("STRING", {"default": '', "multiline": True}),
                "style": ("STRING", {"default": '', "multiline": True}),
                "negative": ("STRING", {"default": '', "multiline": True}),
                "negative_style": ("STRING", {"default": '', "multiline": True}),
            },
            "optional": {
                "replace_1": ("STRING", {"forceInput": True}),
                "replace_2": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "STRING",)
    RETURN_NAMES = ("positive_hires", "negative_hires", "positive", "negative", "prompt",)
    RETURN_NAMES = ("positive", "negative", "prompt",)
    FUNCTION = "get_value"
    CATEGORY = "AE.Tools"

    def get_value(self, clip_hires, clip, find_1, find_2, character, posture, background, style, negative, negative_style, replace_1="", replace_2=""):
        replace_1 = clean_string(replace_1)
        replace_2 = clean_string(replace_2)
        find_1 = clean_string(find_1)
        find_2 = clean_string(find_2)

        character = clean_string(character)
        posture = clean_string(posture)
        background = clean_string(background)
        style = clean_string(style)
        negative = clean_string(negative)
        negative_style = clean_string(negative_style)

        character = is_not_blank_replace(character, find_1, replace_1)
        posture = is_not_blank_replace(posture, find_1, replace_1)
        background = is_not_blank_replace(background, find_1, replace_1)

        character = is_not_blank_replace(character, find_2, replace_2)
        posture = is_not_blank_replace(posture, find_2, replace_2)
        background = is_not_blank_replace(background, find_2, replace_2)

        positive_con, = CLIPTextEncode.encode(self, clip, character)
        positive_con_hires, = CLIPTextEncode.encode(self, clip_hires, character)
        negative_con, = CLIPTextEncode.encode(self, clip, negative_style)
        negative_con_hires, = CLIPTextEncode.encode(self, clip_hires, negative_style)

        if is_not_blank(posture):
            posture_con, = CLIPTextEncode.encode(self, clip, posture)
            posture_con_hires, = CLIPTextEncode.encode(self, clip_hires, posture)
            positive_con, = ConditioningConcat.concat(self, positive_con, posture_con)
            positive_con_hires, = ConditioningConcat.concat(self, positive_con_hires, posture_con_hires)

        if is_not_blank(background) and is_not_blank(style):
            background_con, = CLIPTextEncode.encode(self, clip, background)
            background_con_hires, = CLIPTextEncode.encode(self, clip_hires, background)
            style_con, = CLIPTextEncode.encode(self, clip, style)
            style_con_hires, = CLIPTextEncode.encode(self, clip_hires, style)
            background_style_con, = ConditioningConcat.concat(self, background_con, style_con)
            background_style_con_hires, = ConditioningConcat.concat(self, background_con_hires, style_con_hires)
            positive_con, = ConditioningCombine.combine(self, positive_con, background_style_con)
            positive_con_hires, = ConditioningCombine.combine(self, positive_con_hires, background_style_con_hires)
        elif is_not_blank(background):
            background_con, = CLIPTextEncode.encode(self, clip, background)
            background_con_hires, = CLIPTextEncode.encode(self, clip_hires, background)
            positive_con, = ConditioningCombine.combine(self, positive_con, background_con)
            positive_con_hires, = ConditioningCombine.combine(self, positive_con_hires, background_con_hires)
        elif is_not_blank(style):
            style_con, = CLIPTextEncode.encode(self, clip, style)
            style_con_hires, = CLIPTextEncode.encode(self, clip_hires, style)
            positive_con, = ConditioningCombine.combine(self, positive_con, style_con)
            positive_con_hires, = ConditioningCombine.combine(self, positive_con_hires, style_con_hires)

        if is_not_blank(negative):
            negative_con_base, = CLIPTextEncode.encode(self, clip, negative)
            negative_con_base_hires, = CLIPTextEncode.encode(self, clip_hires, negative)
            negative_con, = ConditioningConcat.concat(self, negative_con_base, negative_con)
            negative_con_hires, = ConditioningConcat.concat(self, negative_con_base_hires, negative_con_hires)
        
        prompt = "\r\n\r\n".join(filter_non_empty_strings([character, posture, background, style, negative, negative_style]))

        return (positive_con_hires, negative_con_hires, positive_con, negative_con, prompt,)
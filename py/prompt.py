import re
from nodes import CLIPTextEncode, ConditioningCombine, ConditioningConcat
from .base import is_blank, is_not_blank, clean_string, is_not_blank_replace

def get_prompt(prompt, positive, background, positive_style, negative, negative_style):
    text = ""

    if is_not_blank(prompt):
        text += f"[Positive]\n{prompt}\n\n"
    if is_not_blank(positive):
        text += f"[Posture]\n{positive}\n\n"
    if is_not_blank(background):
        text += f"[Background]\n{background}\n\n"
    if is_not_blank(positive_style):
        text += f"[Positive Style]\n{positive_style}\n\n"
    if is_not_blank(negative):
        text += f"[Negative]\n{negative}\n\n"
    if is_not_blank(negative_style):
        text += f"[Negative Style]\n{negative_style}\n\n"

    return text

def prepare_conditioning(clip, concat_prompt, positive, posture, background, positive_style, negative, negative_style):
    text_encode = CLIPTextEncode()
    
    if concat_prompt:
        prompt_parts = []
        if is_not_blank(posture):
            prompt_parts.append(posture)
        if is_not_blank(background):
            prompt_parts.append(background)
        if is_not_blank(positive_style):
            prompt_parts.insert(0, positive_style)
        if is_not_blank(negative):
            negative_style = f"{negative_style} ; {negative}" if is_not_blank(negative_style) else negative
        positive = " ; ".join(filter(is_not_blank, [positive] + prompt_parts))

    positive_con, = text_encode.encode(clip, positive)
    negative_con, = text_encode.encode(clip, negative_style)

    if not concat_prompt:
        conditioning_concat = ConditioningConcat()

        if is_not_blank(posture) and is_not_blank(background):
            posture_background_con, = text_encode.encode(clip, f"{posture} ; {background}")
            positive_con, = conditioning_concat.concat(positive_con, posture_background_con)
        elif is_not_blank(posture):
            posture_con, = text_encode.encode(clip, posture)
            positive_con, = conditioning_concat.concat(positive_con, posture_con)
        elif is_not_blank(background):
            background_con, = text_encode.encode(clip, background)
            positive_con, = conditioning_concat.concat(positive_con, background_con)

        if is_not_blank(positive_style):
            style_con, = text_encode.encode(clip, positive_style)
            positive_con, = ConditioningCombine().combine(positive_con, style_con)

        if is_not_blank(negative):
            negative_con_base, = text_encode.encode(clip, negative)
            negative_con, = conditioning_concat.concat(negative_con_base, negative_con)

    return positive_con, negative_con

def process_brackets(text):
    if is_blank(text):
        return text

    def process_round(match):
        tags = match.group(1)
        brackets = match.group(0).count('(') 

        if ':' in tags:
            return match.group(0)

        tags_list = re.split(r'[,\+]+', tags)
        separator = ', ' if ',' in tags else ' + '

        processed_tags = []
        for tag in tags_list:
            tag = tag.strip()
            coefficient = round(1.1 ** brackets, 4)
            processed_tags.append(f"({tag}:{coefficient})")

        return separator.join(processed_tags)

    def process_square(match):
        tags = match.group(1) 
        brackets = match.group(0).count('[')

        if ':' in tags:
            return match.group(0)

        tags_list = re.split(r'[,\+]+', tags)
        separator = ', ' if ',' in tags else ' + '

        processed_tags = []
        for tag in tags_list:
            tag = tag.strip()
            coefficient = round(0.9 ** brackets, 4)
            processed_tags.append(f"({tag}:{coefficient})")

        return separator.join(processed_tags)

    text = re.sub(r'\[+([^\[\]]+)\]+', process_square, text)
    text = re.sub(r'\(+([^()]+)\)+', process_round, text)
    
    return text

POSITIVE = ""
POSITIVE_STYLE = "score_9, score_8_up, score_7_up, high quality, best quality, masterpiece, 4k, 8k"
NEGATIVE = "bad eyes, bad anatomy, bad hands, deformed, ugly, missing fingers, extra fingers"
NEGATIVE_STYLE = "score_6, score_5, score_4, score_3, score_2, score_1, bad quality, lowres, blurry, cropped, jpeg artifacts, mosaic censoring, signature, watermark, artist name"

class SDXLPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "concat_prompt": ("BOOLEAN", {"default": True}),
                "process_tag_weights": ("BOOLEAN", {"default": True}),
                "find_1": ("STRING", {"default": "{x}", "multiline": False}),
                "find_2": ("STRING", {"default": "{y}", "multiline": False}),
                "positive": ("STRING", {"default": POSITIVE, "multiline": True}),
                "posture": ("STRING", {"default": "", "multiline": True}),
                "background": ("STRING", {"default": "", "multiline": True}),
                "positive_style": ("STRING", {"default": POSITIVE_STYLE, "multiline": True}),
                "negative": ("STRING", {"default": NEGATIVE, "multiline": True}),
                "negative_style": ("STRING", {"default": NEGATIVE_STYLE, "multiline": True}),
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

    def get_value(self, clip, concat_prompt, process_tag_weights, find_1, find_2, positive, posture, background, positive_style, negative, negative_style, replace_1="", replace_2=""):
        replace_1 = clean_string(replace_1)
        replace_2 = clean_string(replace_2)
        find_1 = clean_string(find_1)
        find_2 = clean_string(find_2)

        positive = is_not_blank_replace(clean_string(positive), find_1, replace_1)
        posture = is_not_blank_replace(clean_string(posture), find_1, replace_1)
        background = is_not_blank_replace(clean_string(background), find_1, replace_1)
        positive_style = clean_string(positive_style)
        negative = clean_string(negative)
        negative_style = clean_string(negative_style)

        positive = is_not_blank_replace(positive, find_2, replace_2)
        posture = is_not_blank_replace(posture, find_2, replace_2)
        background = is_not_blank_replace(background, find_2, replace_2)

        if process_tag_weights:
            positive = process_brackets(positive)
            posture = process_brackets(posture)
            background = process_brackets(background)
            positive_style = process_brackets(positive_style)
            negative = process_brackets(negative)
            negative_style = process_brackets(negative_style)

        prompt_text = get_prompt(positive, posture, background, positive_style, negative, negative_style)

        positive_con, negative_con = prepare_conditioning(
            clip, concat_prompt, positive, posture, background, positive_style, negative, negative_style
        )

        return (positive_con, negative_con, prompt_text,)


class SDXLPromptWithHires:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_hires": ("CLIP",),
                "concat_prompt": ("BOOLEAN", {"default": True}),
                "process_tag_weights": ("BOOLEAN", {"default": True}),
                "find_1": ("STRING", {"default": "{x}", "multiline": False}),
                "find_2": ("STRING", {"default": "{y}", "multiline": False}),
                "positive": ("STRING", {"default": POSITIVE, "multiline": True}),
                "posture": ("STRING", {"default": "", "multiline": True}),
                "background": ("STRING", {"default": "", "multiline": True}),
                "positive_style": ("STRING", {"default": POSITIVE_STYLE, "multiline": True}),
                "negative": ("STRING", {"default": NEGATIVE, "multiline": True}),
                "negative_style": ("STRING", {"default": NEGATIVE_STYLE, "multiline": True}),
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

    def get_value(self, clip, clip_hires, concat_prompt, process_tag_weights, find_1, find_2, positive, posture, background, positive_style, negative, negative_style, replace_1="", replace_2=""):
        replace_1 = clean_string(replace_1)
        replace_2 = clean_string(replace_2)
        find_1 = clean_string(find_1)
        find_2 = clean_string(find_2)

        positive = is_not_blank_replace(clean_string(positive), find_1, replace_1)
        posture = is_not_blank_replace(clean_string(posture), find_1, replace_1)
        background = is_not_blank_replace(clean_string(background), find_1, replace_1)
        positive_style = clean_string(positive_style)
        negative = clean_string(negative)
        negative_style = clean_string(negative_style)

        positive = is_not_blank_replace(positive, find_2, replace_2)
        posture = is_not_blank_replace(posture, find_2, replace_2)
        background = is_not_blank_replace(background, find_2, replace_2)
        
        if process_tag_weights:
            positive = process_brackets(positive)
            posture = process_brackets(posture)
            background = process_brackets(background)
            positive_style = process_brackets(positive_style)
            negative = process_brackets(negative)
            negative_style = process_brackets(negative_style)

        prompt_text = get_prompt(positive, posture, background, positive_style, negative, negative_style)

        positive_con, negative_con = prepare_conditioning(
            clip, concat_prompt, positive, posture, background, positive_style, negative, negative_style
        )

        if clip_hires != clip:
            positive_con_hires, negative_con_hires = prepare_conditioning(
                clip_hires, concat_prompt, positive, posture, background, positive_style, negative, negative_style
            )
        else:
            positive_con_hires, negative_con_hires =  positive_con, negative_con

        return (positive_con, negative_con, positive_con_hires, negative_con_hires, prompt_text,)
import os

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
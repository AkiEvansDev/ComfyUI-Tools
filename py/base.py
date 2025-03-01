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
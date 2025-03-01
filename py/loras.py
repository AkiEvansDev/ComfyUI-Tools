import re
import os
import folder_paths

def get_and_strip_loras(prompt, silent=False):
    pattern = '<lora:([^:>]*?)(?::(-?\d*(?:\.\d*)?))?>'
    lora_paths = folder_paths.get_filename_list('loras')

    matches = re.findall(pattern, prompt)

    loras = []
    unfound_loras = []
    skipped_loras = []
    for match in matches:
        tag_path = match[0]

        strength = float(match[1] if len(match) > 1 and len(match[1]) else 1.0)
        if strength == 0:
            skipped_loras.append({'lora': tag_path, 'strength': strength})
            continue

        lora_path = get_lora_by_filename(tag_path, lora_paths)
        if lora_path is None:
            unfound_loras.append({'lora': tag_path, 'strength': strength})
            continue

        loras.append({'lora': lora_path, 'strength': strength})

    return (re.sub(pattern, '', prompt), loras, skipped_loras, unfound_loras)

def get_lora_by_filename(file_path, lora_paths=None):
    lora_paths = lora_paths if lora_paths is not None else folder_paths.get_filename_list('loras')

    if file_path in lora_paths:
        return file_path

    lora_paths_no_ext = [os.path.splitext(x)[0] for x in lora_paths]

    if file_path in lora_paths_no_ext:
        found = lora_paths[lora_paths_no_ext.index(file_path)]
        return found

    file_path_force_no_ext = os.path.splitext(file_path)[0]
    if file_path_force_no_ext in lora_paths_no_ext:
        found = lora_paths[lora_paths_no_ext.index(file_path_force_no_ext)]
        return found

    lora_filenames_only = [os.path.basename(x) for x in lora_paths]
    if file_path in lora_filenames_only:
        found = lora_paths[lora_filenames_only.index(file_path)]
        return found

    file_path_force_filename = os.path.basename(file_path)
    lora_filenames_only = [os.path.basename(x) for x in lora_paths]
    if file_path_force_filename in lora_filenames_only:
        found = lora_paths[lora_filenames_only.index(file_path_force_filename)]
        return found

    lora_filenames_and_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in lora_paths]
    if file_path in lora_filenames_and_no_ext:
        found = lora_paths[lora_filenames_and_no_ext.index(file_path)]
        return found

    file_path_force_filename_and_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    if file_path_force_filename_and_no_ext in lora_filenames_and_no_ext:
        found = lora_paths[lora_filenames_and_no_ext.index(file_path_force_filename_and_no_ext)]
        return found

    for index, lora_path in enumerate(lora_paths):
        if file_path in lora_path:
            found = lora_paths[index]
            return found

    return None

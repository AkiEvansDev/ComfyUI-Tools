import re
import ijson
from collections import defaultdict

#https://huggingface.co/datasets/deepghs/site_tags

def remove_consecutive_duplicates(input_string):
    parts = input_string.split("_")
    
    unique_parts = []
    previous_part = None
    for part in parts:
        if part != previous_part:
            unique_parts.append(part)
            previous_part = part
    
    result = "_".join(unique_parts)
    
    return result

def process_tag(tag):
    tag = re.sub(r'[\r\n\u2028\u2029\u200B]', "", tag)
    tag = re.sub(r'\s+', " ", tag)
    tag = tag.replace("/", " ").replace("\\", " ")

    tag = re.sub(r'(?<!^)([A-Z])', r' \1', tag)
    tag = re.sub(r'(\()\s*', r' \1', tag)
    tag = re.sub(r'\s*(\))', r'\1', tag)
    
    def process_inside_brackets(match):
        content = match.group(1)
        content = re.sub(r'(?<!^)([A-Z])', r' \1', content)
        content = content.lower().replace(" ", "_")
        return f"({content})"
    
    tag = re.sub(r'\((.*?)\)', process_inside_brackets, tag)
    tag = tag.lower().replace(" ", "_")

    tag = tag.replace(".", "").replace(":", "")
    tag = re.sub(r'_+', '_', tag)
    tag = tag.strip("_ ")

    if not re.search(r'[a-zA-Z]', tag):
        return "-"
    
    return remove_consecutive_duplicates(tag)

def read_large_json(file_path, item_key):
    with open(file_path, 'r', encoding='utf-8') as file:
        for item in ijson.items(file, item_key):
            yield item

def read_autocomplete_old(file_path):
    tags = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tag, count = line.strip().split(',')
            tags[process_tag(tag)] = int(count)

    return tags

def read_json_tags(file_path, item_key, tag_field, count_field):
    tags = defaultdict(int)
    found_any = False

    for item in read_large_json(file_path, item_key):
        try:
            tag = process_tag(item[tag_field])
            count = item[count_field]
            tags[tag] += int(count)
            if int(count) >= 100:
                found_any = True
        except KeyError as e:
            print(f"Error: File {file_path} not contains {e}.")
            continue

    if not found_any:
        print(f"Warning: File {file_path} not contains tag with count >= 100.")

    return tags

def write_autocomplete(file_path, tags):
    with open(file_path, 'w', encoding='utf-8') as file:
        for tag, count in sorted(tags.items(), key=lambda x: x[1], reverse=True):
            if count >= 100 and tag != "-":
                file.write(f"{tag},{count}\n")

def main():
    tags = defaultdict(int)

    autocomplete_old = read_autocomplete_old('autocomplete_old.txt')
    for tag, count in autocomplete_old.items():
        tags[tag] = count

    json_files = [
        ('anime-pictures.net.tags.json', 'item', 'tag', 'num_pub'),
        ('booru.allthefallen.moe.tags.json', 'item', 'name', 'post_count'),
        ('chan.sankakucomplex.com.tags.json', 'item', 'name', 'post_count'),
        ('danbooru.donmai.us.tags.json', 'item', 'name', 'post_count'),
        ('gelbooru.com.tags.json', 'item', 'name', 'count'),
        ('hypnohub.net.tags.json', 'item', 'name', 'count'),
        ('konachan.com.tags.json', 'item', 'name', 'count'),
        ('konachan.net.tags.json', 'item', 'name', 'count'),
        ('lolibooru.moe.tags.json', 'item', 'name', 'post_count'),
        ('rule34.xxx.tags.json', 'item', 'name', 'count'),
        ('safebooru.donmai.us.tags.json', 'item', 'name', 'post_count'),
        ('wallhaven.cc.tags.json', 'item', 'name', 'posts'),
        ('xbooru.com.tags.json', 'item', 'name', 'count'),
        ('yande.re.tags.json', 'item', 'name', 'count'),
    ]

    for file_path, item_key, tag_field, count_field in json_files:
        json_tags = read_json_tags(file_path, item_key, tag_field, count_field)
        for tag, count in json_tags.items():
            tags[tag] += count

    write_autocomplete('autocomplete.txt', tags)

if __name__ == "__main__":
    main()
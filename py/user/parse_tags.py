import json
import ijson
from collections import defaultdict

#https://huggingface.co/datasets/deepghs/site_tags/tree/df6157d81730204e710976409227d8c845a9f1f4

def read_large_json(file_path, item_key):
    with open(file_path, 'r', encoding='utf-8') as file:
        for item in ijson.items(file, item_key):
            yield item

def read_autocomplete_old(file_path):
    tags = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tag, count = line.strip().split(',')
            tags[tag] = int(count)
    return tags

def write_autocomplete(file_path, tags):
    with open(file_path, 'w', encoding='utf-8') as file:
        for tag, count in sorted(tags.items(), key=lambda x: x[1], reverse=True):
            file.write(f"{tag},{count}\n")

def main():
    tags = defaultdict(int)

    anime_pictures = read_large_json('anime-pictures.tags.json', 'item')
    for item in anime_pictures:
        tag = item['tag']
        count = item['num_pub']
        if count >= 100:
            tags[tag] = max(tags[tag], count)

    sankakucomplex = read_large_json('chan.sankakucomplex.tags.json', 'item')
    for item in sankakucomplex:
        tag = item['name']
        count = item['post_count']
        if count >= 100:
            tags[tag] = max(tags[tag], count)

    danbooru = read_large_json('danbooru.donmai.tags.json', 'item')
    for item in danbooru:
        tag = item['name']
        count = item['post_count']
        if count >= 100:
            tags[tag] = max(tags[tag], count)

    zerochan = read_large_json('zerochan.tags.json', 'item')
    for item in zerochan:
        tag = item['tag']
        count = item['total']
        if count >= 100:
            tags[tag] = max(tags[tag], count)

    autocomplete_old = read_autocomplete_old('autocomplete_old.txt')
    for tag, count in autocomplete_old.items():
        if count >= 100:
            tags[tag] = max(tags[tag], count)

    write_autocomplete('autocomplete.txt', tags)

if __name__ == "__main__":
    main()
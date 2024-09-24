from typing import List

import ujson as json


def remove_annotation_dropout(annotation_path: str) -> (str, int):
    # Annotation is jsonl
    with open(annotation_path, 'r') as f:
        annotations = [json.loads(line) for line in f]

    new_annotations = []
    for annotation in annotations:
        if '<image>\n<audio>\n' in annotation['conversations'][0]['value'] and \
                'The speaker said' in annotation['conversations'][0]['value']:
            new_annotations.append(annotation)

    new_path = annotation_path.replace('.jsonl', '_nodrop.jsonl')
    with open(new_path, 'w') as f:
        for annotation in new_annotations:
            f.write(json.dumps(annotation) + '\n')

    return new_path, len(new_annotations)


if __name__ == '__main__':
    meta_path = 'shell/data/mental_health_ft.json'
    meta = json.load(open(meta_path, 'r'))
    for dataset_name, ds_collections in meta.items():
        anno_path = ds_collections['annotation']
        new_path, new_length = remove_annotation_dropout(anno_path)
        print(f'New annotation path: {new_path}')
        meta[dataset_name]['annotation'] = new_path
        meta[dataset_name]['length'] = new_length

    new_meta_path = meta_path.replace('.json', '_nodrop.json')
    with open(new_meta_path, 'w') as f:
        json.dump(meta, f, indent=4)

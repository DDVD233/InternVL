import os

import ujson as json
from PIL import Image, ImageDraw, ImageFont
import tqdm


def visualize_dataset(annotation_path: str, root_path: str):
    with open(annotation_path, 'r') as f:
        annotations = [json.loads(line) for line in f]

    for annotation in tqdm.tqdm(annotations):
        if 'image' not in annotation or 'audio' not in annotation:
            continue
        image_path = annotation['image']
        absolute_image_path = f'{root_path}/{image_path}'
        question = annotation['conversations'][0]['value']
        target = annotation['conversations'][1]['value']

        # Open the image
        with Image.open(absolute_image_path) as img:
            # Choose a font (the default font or provide a path to a .ttf file)
            font = ImageFont.load_default()

            # Define the size of the extension for the canvas based on the text
            text = f"Q: {question}\nA: {target}"
            draw = ImageDraw.Draw(img)
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]  # Get the height from the bounding box

            # Calculate new image size
            new_height = img.height + text_height + 20  # Adding some padding
            new_img = Image.new('RGB', (img.width, new_height), "white")
            new_img.paste(img, (0, 0))

            # Add text to the new image
            draw = ImageDraw.Draw(new_img)
            draw.text((10, img.height + 10), text, fill="black", font=font)

            # Save the new image
            new_img_path = f"{root_path}/modified_{image_path}"
            if not os.path.exists(os.path.dirname(new_img_path)):
                os.makedirs(os.path.dirname(new_img_path))
            new_img.save(new_img_path)


if __name__ == '__main__':
    visualize_dataset('/home/dvd/data/datasets/cmu_mosei/annotation_train_diversified.jsonl',
                      '/home/dvd/data/datasets/cmu_mosei/')
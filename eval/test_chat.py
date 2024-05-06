import os

import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, CLIPProcessor
from transformers import AutoTokenizer
from datasets.IemocapDataset import IemocapDataset
from internvl.model.internvl_chat import InternVLChatModel


def make_grid(images):
    # Reshape the input tensor to (B, 2, 2, H, W, C)
    images = images.view(images.shape[0], 2, 2, images.shape[2], images.shape[3], images.shape[4])

    # Swap the second and third dimensions to move the images next to each other
    images = images.permute(0, 3, 1, 4, 2, 5)

    # Reshape to (B, 2H, 2W, C)
    images = images.contiguous().view(images.shape[0], images.shape[3] * 2, images.shape[1] * 2, images.shape[5])

    return images


def main():
    # Initialize IemocapDataset
    iemocap_dataset = IemocapDataset(
        '/home/dvd/data/depression_interview/dataset/IEMOCAP_full_release/IEMOCAP_full_release')
    data_loader = torch.utils.data.DataLoader(iemocap_dataset, batch_size=1, shuffle=True)
    path = "OpenGVLab/InternViT-6B-448px-V1-5"
    model = InternVLChatModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(path)
    image_processor = CLIPImageProcessor.from_pretrained(path)

    # Iterate over data
    output_path = '/home/dvd/data/depression_interview/iemocap_output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for index, sample in enumerate(data_loader):
        frames: torch.Tensor = sample['frames'][0]  # 4, H, W, C
        # make 2x2 grid of frames, shape = 2H, 2W, C
        # frame = frames[0]
        # grid = make_grid(frames)

        # for index_f, frame in enumerate(frames):
        pixel_values = image_processor(images=frames, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        question = ("What is the emotion of the speaker? "
                    "Answer with one word from the following: happy, sad, neutral, angry, excited, and frustrated.")
        response = model.multi_image_chat(tokenizer, pixel_values, 4, question, generation_config)
        print('-' * 50)
        print(response)
        print('-' * 50)
        # save image to output_path
        grid = make_grid(frames)
        img = Image.fromarray(grid.cpu().numpy())
        img.save(os.path.join(output_path, f'{index}.jpg'))
        # save response to output_path
        with open(os.path.join(output_path, f'{index}.txt'), 'w') as file:
            file.write(response)


if __name__ == '__main__':
    main()
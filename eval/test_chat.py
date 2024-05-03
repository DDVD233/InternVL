import os

import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, CLIPProcessor
from transformers import AutoTokenizer
from datasets.IemocapDataset import IemocapDataset
from torchvision.utils import save_image

# Initialize IemocapDataset
iemocap_dataset = IemocapDataset('/home/dvd/data/depression_interview/dataset/IEMOCAP_full_release/IEMOCAP_full_release')
path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
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

for index, sample in enumerate(iemocap_dataset):
    frames: torch.Tensor = sample['frames']
    for index_f, frame in enumerate(frames):
        pixel_values = image_processor(images=frame, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        question = "Please describe the image."
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        print('-' * 50)
        print(response)
        print('-' * 50)
        # save image to output_path
        image = save_image(frame, os.path.join(output_path, f'{index}_{index_f}.jpg'))
        # save response to output_path
        with open(os.path.join(output_path, f'{index}_{index_f}.txt'), 'w') as file:
            file.write(response)




# path = "OpenGVLab/InternVL-Chat-V1-1"
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True).eval().cuda()
#
# tokenizer = AutoTokenizer.from_pretrained(path)
# image = Image.open('./examples/image2.jpg').convert('RGB')
# image = image.resize((448, 448))
# image_processor = CLIPImageProcessor.from_pretrained(path)
#
# pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
# pixel_values = pixel_values.to(torch.bfloat16).cuda()
#
# generation_config = dict(
#     num_beams=1,
#     max_new_tokens=512,
#     do_sample=False,
# )
#
# question = "Please describe the image."
# response = model.chat(tokenizer, pixel_values, question, generation_config)

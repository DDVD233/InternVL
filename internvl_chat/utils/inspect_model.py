import os

from safetensors import safe_open


def inspect_model():
    # Path to the multipart .safetensors file
    model_dir = '/home/dvd/data/outputs/phq9_8B_lora_nodrop_split1'

    for files in os.listdir(model_dir):
        if files.endswith('.safetensors'):
            file_path = os.path.join(model_dir, files)

            # Open the file and list all parameter names
            with safe_open(file_path, framework="pt") as f:
                parameter_names = list(f.keys())

            # Print the parameter names
            for param_name in parameter_names:
                print(f"Parameter name: {param_name}")


if __name__ == '__main__':
    inspect_model()

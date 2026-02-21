import torch
import folder_paths
import os
from . import networks  # networks.py copied from junyanz/pytorch-CycleGAN-and-pix2pix

folder_paths.add_model_folder_path("pix2pix", os.path.join(folder_paths.models_dir, "pix2pix"))

class Pix2PixLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": (folder_paths.get_filename_list("pix2pix"),),
                "netG": (["unet_256", "unet_128", "resnet_9blocks", "resnet_6blocks"],),
                "input_nc": ("INT", {"default": 3, "min": 1, "max": 4}),
                "output_nc": ("INT", {"default": 3, "min": 1, "max": 4}),
                "ngf": ("INT", {"default": 64, "min": 16, "max": 256}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "My Custom Nodes"

    def run_inference(self, image, model_path, netG, input_nc, output_nc, ngf):
        # A. Setup Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # B. Build Generator using the original repo's factory function
        model = networks.define_G(input_nc, output_nc, ngf, netG, norm="batch",
                                  use_dropout=False, init_type="normal", init_gain=0.02)

        # C. Load Weights
        # The original repo saves state dicts directly via torch.save(net.state_dict(), path)
        full_path = folder_paths.get_full_path("pix2pix", model_path)
        state_dict = torch.load(full_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # D. Pre-process Image (CRITICAL STEP)
        # ComfyUI sends images as [Batch, Height, Width, Channels] (BHWC)
        # PyTorch models usually expect [Batch, Channels, Height, Width] (BCHW)
        input_tensor = image.permute(0, 3, 1, 2).to(device)
        
        # Pix2Pix usually expects normalization between -1 and 1. 
        # ComfyUI gives 0 to 1.
        input_tensor = (input_tensor * 2.0) - 1.0

        # E. Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # F. Post-process Image
        # Convert back from -1..1 to 0..1
        output_tensor = (output_tensor + 1.0) / 2.0
        # Clamp to ensure valid colors
        output_tensor = torch.clamp(output_tensor, 0, 1)
        # Convert back to BHWC for ComfyUI
        result = output_tensor.permute(0, 2, 3, 1).cpu()

        return (result,)

# Register the node
NODE_CLASS_MAPPINGS = {
    "Pix2PixLoader": Pix2PixLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pix2PixLoader": "Pix2Pix GAN"
}
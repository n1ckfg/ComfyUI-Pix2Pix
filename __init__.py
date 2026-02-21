import torch
import folder_paths
import os
import platform
import numpy as np
from . import networks  # networks.py copied from junyanz/pytorch-CycleGAN-and-pix2pix

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

folder_paths.add_model_folder_path("pix2pix", os.path.join(folder_paths.models_dir, "pix2pix"))


def get_onnx_providers():
    """Return ONNX Runtime execution providers in priority order for the current platform.

    Provider selection:
      macOS   -> CoreML (Apple Neural Engine / GPU) -> CPU
      Windows -> CUDA -> DirectML (AMD/Intel/NVIDIA via DX12) -> CPU
      Linux   -> CUDA -> ROCm (AMD) -> CPU
      other   -> CPU
    Only providers reported as available by onnxruntime are included.
    """
    available = ort.get_available_providers()
    providers = []
    system = platform.system()

    if system == "Darwin":
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
    elif system == "Windows":
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
    else:  # Linux and anything else
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "ROCMExecutionProvider" in available:
            providers.append("ROCMExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


class Pix2PixLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ([f for f in folder_paths.get_filename_list("pix2pix") if f != ".DS_Store"],),
                # The following four parameters are only used for .pth models.
                # They are ignored when an .onnx model is selected.
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
        full_path = folder_paths.get_full_path("pix2pix", model_path)
        if model_path.lower().endswith(".onnx"):
            return self._run_onnx(image, full_path)
        else:
            return self._run_pytorch(image, full_path, netG, input_nc, output_nc, ngf)

    # ------------------------------------------------------------------
    # PyTorch (.pth) path — unchanged from original implementation
    # ------------------------------------------------------------------

    def _run_pytorch(self, image, full_path, netG, input_nc, output_nc, ngf):
        # A. Setup Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # B. Build Generator using the original repo's factory function
        model = networks.define_G(input_nc, output_nc, ngf, netG, norm="batch",
                                  use_dropout=False, init_type="normal", init_gain=0.02)

        # C. Load Weights
        # The original repo saves state dicts directly via torch.save(net.state_dict(), path)
        state_dict = torch.load(full_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # D. Pre-process: BHWC -> BCHW, [0,1] -> [-1,1]
        input_tensor = image.permute(0, 3, 1, 2).to(device)
        input_tensor = (input_tensor * 2.0) - 1.0

        # E. Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # F. Post-process: [-1,1] -> [0,1], BCHW -> BHWC
        output_tensor = (output_tensor + 1.0) / 2.0
        output_tensor = torch.clamp(output_tensor, 0, 1)
        result = output_tensor.permute(0, 2, 3, 1).cpu()

        return (result,)

    # ------------------------------------------------------------------
    # ONNX Runtime path — cross-platform GPU acceleration
    # ------------------------------------------------------------------

    def _run_onnx(self, image, full_path):
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed. Install one of:\n"
                "  pip install onnxruntime              # CPU only\n"
                "  pip install onnxruntime-gpu          # NVIDIA CUDA\n"
                "  pip install onnxruntime-directml     # Windows DirectML (AMD/Intel/NVIDIA)\n"
                "CoreML (macOS) is included in the base onnxruntime package."
            )

        providers = get_onnx_providers()
        print(f"[Pix2Pix] ONNX providers: {providers}")

        session = ort.InferenceSession(full_path, providers=providers)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Pre-process: BHWC -> BCHW, [0,1] -> [-1,1]
        input_array = image.permute(0, 3, 1, 2).numpy().astype(np.float32)
        input_array = (input_array * 2.0) - 1.0

        # Inference
        output_array = session.run([output_name], {input_name: input_array})[0]

        # Post-process: [-1,1] -> [0,1], BCHW -> BHWC
        output_array = (output_array + 1.0) / 2.0
        output_array = np.clip(output_array, 0, 1)
        result = torch.from_numpy(output_array).permute(0, 2, 3, 1)

        return (result,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "Pix2PixLoader": Pix2PixLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pix2PixLoader": "Pix2Pix GAN"
}

import os
import time
import requests
import tempfile
import json
import numpy as np
import torch
from io import BytesIO
from PIL import Image
import replicate

class ReplicateAPI_flux_fill_pro:
    @classmethod
    def INPUT_TYPES(cls):
        """
        A single synchronous generation node for `black-forest-labs/flux-fill-pro`.
        
        Required fields:
          - api_token (STRING)
          - prompt (STRING)
          - steps (INT)
          - guidance (FLOAT)
          - outpaint (STRING)
          - output_format (choice of "jpg" or "png")
          - safety_tolerance (INT)

        Optional fields (ComfyUI image inputs):
          - mask (IMAGE)
          - image (IMAGE)
        """
        return {
            "required": {
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "display": "Replicate API Token"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "movie poster"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "display": "Steps"
                }),
                "guidance": ("FLOAT", {
                    "default": 3.0,
                    "min": 0,
                    "max": 20,
                    "step": 0.1,
                    "display": "Guidance"
                }),
                "outpaint": ("STRING", {
                    "default": "Zoom out 2x",
                    "display": "Outpaint Type"
                }),
                "output_format": (["jpg", "png"], {
                    "default": "jpg",
                    "display": "Output Format"
                }),
                "safety_tolerance": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 10,
                    "display": "Safety Tolerance"
                }),
            },
            "optional": {
                "mask": ("IMAGE", {
                    "display": "Mask Image (Optional)"
                }),
                "image": ("IMAGE", {
                    "display": "Source Image (Optional)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "generation_info",)
    FUNCTION = "generate"
    CATEGORY = "Replicate"

    def __init__(self):
        # Output directories for saving images & metadata
        self.output_dir = "output/API/Replicate/flux-fill-pro-single"
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def get_next_number(self):
        """
        Find the next available integer filename (e.g. 001.jpg).
        """
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if not files:
            return 1

        numbers = []
        for f in files:
            base, ext = os.path.splitext(f)
            try:
                numbers.append(int(base))
            except ValueError:
                pass
        if numbers:
            return max(numbers) + 1
        else:
            return 1

    def create_filename(self, number, ext="jpg"):
        """
        Format the filename like 001.jpg or 002.png, etc.
        """
        return f"{number:03d}.{ext}"

    def save_image_and_metadata(self, pil_img, generation_info, number, ext="jpg"):
        """
        Saves the PIL image and the metadata JSON file. Returns file paths.
        """
        file_basename = self.create_filename(number, ext)
        image_path = os.path.join(self.output_dir, file_basename)

        # Save the image
        pil_img.save(image_path, format=ext.upper())

        # Save JSON metadata
        meta_filename = f"{number:03d}_metadata.json"
        meta_path = os.path.join(self.metadata_dir, meta_filename)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(generation_info, f, indent=4, ensure_ascii=False)

        return image_path, meta_path

    def generate(
        self,
        api_token,
        prompt,
        steps,
        guidance,
        outpaint,
        output_format,
        safety_tolerance,
        mask=None,
        image=None
    ):
        """
        Synchronous single-image generation for flux-fill-pro.
        No multi-image or seeds. Just a single replicate.run() call.

        ComfyUI calls this function. We:

        1) Convert 'mask' and 'image' from ComfyUI tensors to temp files (if provided).
        2) Call replicate.run("black-forest-labs/flux-fill-pro", ...)
        3) Download the result
        4) Save image & metadata
        5) Return (IMAGE, STRING)
        """
        # Validate
        if not api_token:
            raise ValueError("API token is required for Replicate calls.")

        # We'll return a placeholder if generation fails
        empty_img = torch.zeros((1, 512, 512, 3))

        try:
            # 1) Build the input dictionary
            os.environ["REPLICATE_API_TOKEN"] = api_token

            input_dict = {
                "prompt": prompt,
                "steps": steps,
                "guidance": guidance,
                "outpaint": outpaint,
                "output_format": output_format,
                "safety_tolerance": safety_tolerance,
                "prompt_upsampling": True,  # Set as default in the code instead of input
            }

            # 2) Convert mask/image to local files if given
            mask_file = None
            image_file = None

            if mask is not None:
                mask_file = self.tensor_to_tempfile(mask, suffix=".png")
                input_dict["mask"] = mask_file

            if image is not None:
                image_file = self.tensor_to_tempfile(image, suffix=".png")
                input_dict["image"] = image_file

            # 3) Call replicate.run (sync)
            output = replicate.run("black-forest-labs/flux-fill-pro", input=input_dict)

            # Clean up temp files
            if mask_file is not None:
                mask_file.close()
                os.remove(mask_file.name)
            if image_file is not None:
                image_file.close()
                os.remove(image_file.name)

            if not output:
                raise ValueError("No valid result from replicate.run().")

            # Print output for debugging
            print(f"Replicate output: {output}")

            # Handle different output formats
            image_url = None
            if isinstance(output, dict) and 'image' in output:
                image_url = output['image']
            elif isinstance(output, list) and len(output) > 0:
                image_url = output[0]
            elif isinstance(output, str):
                image_url = output
            else:
                raise ValueError(f"Unexpected output format from replicate: {output}")

            if not image_url:
                raise ValueError("No image URL found in replicate output")

            # Download the image
            resp = requests.get(image_url)
            if resp.status_code != 200:
                raise ConnectionError(f"Failed to download result image. Status code: {resp.status_code}")

            try:
                pil_img = Image.open(BytesIO(resp.content))
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to process downloaded image: {str(e)}")

            # Save image and metadata
            number = self.get_next_number()

            # We'll remove any non-serializable items from input_dict
            safe_dict = dict(input_dict)
            safe_dict.pop("mask", None)
            safe_dict.pop("image", None)

            generation_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "black-forest-labs/flux-fill-pro",
                "parameters": safe_dict,
                "replicate_output": str(output),
            }

            image_path, meta_path = self.save_image_and_metadata(pil_img, generation_info, number, ext=output_format)
            print(f"[flux-fill-pro] Single generation: saved image -> {image_path}")
            print(f"[flux-fill-pro] Single generation: saved metadata -> {meta_path}")

            # Convert to ComfyUI torch tensor (B, H, W, C)
            result_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            result_tensor = result_tensor.unsqueeze(0)  # Add batch dimension

            return (result_tensor, json.dumps(generation_info, indent=2))

        except Exception as e:
            error_info = {
                "error": f"Flux-fill-pro single generation failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            return (empty_img, json.dumps(error_info, indent=2))

    def tensor_to_tempfile(self, tensor, suffix=".png"):
        """
        Convert a ComfyUI image tensor to a local file object, so we can pass it to replicate.
        We'll return an open file in "rb" mode, and the caller is responsible for closing/deleting it.
        """
        pil_img = self.tensor_to_pil(tensor)
        fd, filename = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        pil_img.save(filename, format=suffix.upper().lstrip("."))
        return open(filename, "rb")

    def tensor_to_pil(self, tensor):
        """
        Convert (B, H, W, C) or (H, W, C) or (C, H, W) to a PIL image. 
        Typically, ComfyUI uses (B, H, W, C). We'll handle that.
        """
        # If the tensor is 4D, remove batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        # Convert to numpy array
        arr = tensor.cpu().numpy()
        
        # If shape is (C, H, W), transpose to (H, W, C)
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1, 2, 0))

        # Scale to 0-255 range and convert to uint8
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        ComfyUI-specific method to handle caching logic. 
        Returning NaN tells ComfyUI there's no caching for this node.
        """
        return float("NaN")

    def interrupt(self):
        """
        For multi-step processes, we could handle an interrupt event, 
        but here it's just a single call. 
        We'll keep it for completeness.
        """
        print("[flux-fill-pro-single] Interrupt: not used in single-image mode.")

NODE_CLASS_MAPPINGS = {
    "ReplicateAPI_flux_fill_pro": ReplicateAPI_flux_fill_pro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplicateAPI_flux_fill_pro": "Replicate Flux-Fill-Pro"
}

__all__ = ["ReplicateAPI_flux_fill_pro"]

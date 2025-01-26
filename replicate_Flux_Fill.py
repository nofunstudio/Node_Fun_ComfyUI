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
        A single-image, synchronous node for `black-forest-labs/flux-fill-pro`.
        The parameters come directly from the model docs, plus
        two optional IMAGE inputs for 'mask' and 'image'.
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
                    "display": "Outpaint"
                }),
                "output_format": (["png", "jpg"], {
                    "default": "png",
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
                # Optional IMAGE inputs from ComfyUI (e.g. Load Image node)
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
        # Same directory structure as your flux ultra node:
        self.output_dir = "output/API/Replicate/flux-fill-pro"
        self.metadata_dir = os.path.join(self.output_dir, "metadata")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def get_next_number(self):
        """
        Looks at existing .png/.jpg files and picks the next integer
        filename index (e.g. 001, 002, etc.).
        """
        valid_exts = {".png", ".jpg"}
        files = [f for f in os.listdir(self.output_dir) 
                 if os.path.splitext(f)[1].lower() in valid_exts]
        if not files:
            return 1

        numbers = []
        for file_name in files:
            base, _ = os.path.splitext(file_name)
            try:
                numbers.append(int(base))
            except ValueError:
                pass
        
        if numbers:
            return max(numbers) + 1
        else:
            return 1

    def create_filename(self, number):
        """
        The flux ultra node used zero-padded filenames ending in .png
        (e.g., '001.png'). We'll keep that exact style here.
        """
        return f"{number:03d}.png"

    def save_image_and_metadata(self, img, generation_info, number):
        """
        Saves the image as .png, plus metadata as a .json.
        Identical to the flux ultra approach.
        """
        filename = self.create_filename(number)
        filepath = os.path.join(self.output_dir, filename)

        # Save image
        img.save(filepath, format="PNG")

        # Create metadata filename (001_metadata.json)
        metadata_filename = f"{number:03d}_metadata.json"
        metadata_filepath = os.path.join(self.metadata_dir, metadata_filename)

        # Write out metadata
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(generation_info, f, indent=4, ensure_ascii=False)

        return filepath, metadata_filepath

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
        Single synchronous call to replicate.run("black-forest-labs/flux-fill-pro").
        Saves image & metadata exactly as done previously, then returns (IMAGE, STRING).
        """
        # For errors, we return an empty 1024x1024 (B=1,H=1024,W=1024,C=3), 
        # which is what your flux ultra node used as a fallback.
        empty_image = torch.zeros((1, 1024, 1024, 3))

        # Make sure we have an API token
        if not api_token:
            raise ValueError("A Replicate API token is required.")

        try:
            # 1) Build the input dictionary
            os.environ["REPLICATE_API_TOKEN"] = api_token
            input_data = {
                "prompt": prompt,
                "steps": steps,
                "guidance": guidance,
                "outpaint": outpaint,
                "output_format": output_format,  # "png" or "jpg"
                "safety_tolerance": safety_tolerance,
                "prompt_upsampling": True,
            }

            # 2) Convert optional mask/image to local files if present
            mask_file = None
            image_file = None

            if mask is not None:
                mask_file = self.tensor_to_tempfile(mask)
                input_data["mask"] = mask_file
            if image is not None:
                image_file = self.tensor_to_tempfile(image)
                input_data["image"] = image_file

            # 3) Call replicate.run() 
            output = replicate.run("black-forest-labs/flux-fill-pro", input=input_data)

            # 4) Clean up local files
            if mask_file is not None:
                mask_file.close()
                os.remove(mask_file.name)
            if image_file is not None:
                image_file.close()
                os.remove(image_file.name)

            if not output:
                raise ValueError("No valid result from replicate.run().")

            # 5) Usually a single URL or a list
            if isinstance(output, list):
                image_url = output[0]
            else:
                image_url = output

            # 6) Download final image
            resp = requests.get(image_url)
            if resp.status_code != 200:
                raise ConnectionError(f"Failed to download image. HTTP {resp.status_code}")

            pil_img = Image.open(BytesIO(resp.content))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # 7) Save the image & metadata with the same flux ultra logic
            number = self.get_next_number()
            # We'll remove non-serializable items from input_data
            safe_input_data = dict(input_data)
            safe_input_data.pop("mask", None)
            safe_input_data.pop("image", None)

            # Keep replicate output as string
            generation_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": safe_input_data,
                "replicate_output": str(output),
                "model": "black-forest-labs/flux-fill-pro"
            }

            image_path, metadata_path = self.save_image_and_metadata(pil_img, generation_info, number)
            print(f"[flux-fill-pro single] Saved image -> {image_path}")
            print(f"[flux-fill-pro single] Saved metadata -> {metadata_path}")

            # 8) Convert to ComfyUI's IMAGE format (1, H, W, 3)
            img_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            img_tensor = img_tensor.unsqueeze(0)

            # 9) Return the image tensor & metadata JSON string
            return (img_tensor, json.dumps(generation_info, indent=2))

        except Exception as e:
            # Return an empty image and an error message in JSON
            error_info = {
                "error": f"Flux-fill-pro single generation failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            return (empty_image, json.dumps(error_info, indent=2))

    def tensor_to_tempfile(self, tensor):
        """
        Matches the flux ultra logic:
        - Convert a ComfyUI IMAGE tensor to a PNG file 
        - Return an open file in "rb" mode (caller must close/delete).
        """
        pil_img = self.tensor_to_pil(tensor)
        fd, filename = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil_img.save(filename, format="PNG")
        return open(filename, "rb")

    def tensor_to_pil(self, tensor):
        """
        Identical to your flux ultra approach: handle (B, H, W, C) or (C, H, W).
        """
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # remove batch dimension

        arr = tensor.cpu().numpy()
        # If shape is (C, H, W), transpose it
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1, 2, 0))

        arr = (arr * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Same as your flux ultra node: 
        returning NaN means no caching for this node.
        """
        return float("NaN")

    def interrupt(self):
        """
        In single-image synchronous logic, there's no partial progress to interrupt. 
        We'll leave it here for consistency with the flux ultra node.
        """
        print("[flux-fill-pro single] Interrupt called (not used in single-image mode).")


# Register with ComfyUI
NODE_CLASS_MAPPINGS = {
    "ReplicateAPI_flux_fill_pro": ReplicateAPI_flux_fill_pro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplicateAPI_flux_fill_pro": "Replicate Flux-Fill-Pro"
}

__all__ = ["ReplicateAPI_flux_fill_pro"]

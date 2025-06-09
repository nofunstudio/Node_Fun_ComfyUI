import os
import time
import requests
import tempfile
import json
import numpy as np
import torch
from io import BytesIO
from PIL import Image
import fal_client

class FalAPI_recraft_upscale:
    @classmethod
    def INPUT_TYPES(cls):
        """
        A single-image node for `fal-ai/recraft/upscale/crisp`.
        Takes an image input and upscales it using Recraft's crisp upscale.
        """
        return {
            "required": {
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "display": "FAL API Token"
                }),
                "image": ("IMAGE", {
                    "display": "Source Image"
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": False,
                    "display": "Enable Safety Checker"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("upscaled_image", "generation_info",)
    FUNCTION = "upscale"
    CATEGORY = "FAL"

    def __init__(self):
        # Directory structure for saving upscaled images
        self.output_dir = "output/API/FAL/recraft-upscale"
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
        Zero-padded filenames ending in .png (e.g., '001.png').
        """
        return f"{number:03d}.png"

    def save_image_and_metadata(self, img, generation_info, number):
        """
        Saves the upscaled image as .png, plus metadata as a .json.
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

    def on_queue_update(self, update):
        """Handle queue updates and log messages"""
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"[Recraft Upscale] {log['message']}")

    def upscale(self, api_token, image, enable_safety_checker):
        """
        Upscale image using fal-ai/recraft/upscale/crisp
        """
        # For errors, return an empty 1024x1024 image
        empty_image = torch.zeros((1, 1024, 1024, 3))

        # Make sure we have an API token
        if not api_token:
            raise ValueError("A FAL API token is required.")

        try:
            # 1) Set up FAL client with API token
            os.environ["FAL_KEY"] = api_token

            # 2) Convert ComfyUI image tensor to temporary file
            image_file = self.tensor_to_tempfile(image)
            
            # 3) Upload image to FAL storage and get URL
            with open(image_file.name, "rb") as f:
                image_url = fal_client.upload_file(f)

            # 4) Build the input arguments
            arguments = {
                "image_url": image_url,
                "enable_safety_checker": enable_safety_checker
            }

            print(f"[Recraft Upscale] Starting upscale with safety_checker={enable_safety_checker}")

            # 5) Call fal_client.subscribe() 
            result = fal_client.subscribe(
                "fal-ai/recraft/upscale/crisp",
                arguments=arguments,
                with_logs=True,
                on_queue_update=self.on_queue_update,
            )

            # 6) Clean up temporary file
            image_file.close()
            os.remove(image_file.name)

            if not result or not result.get("image"):
                raise ValueError("No valid result from fal_client.subscribe().")

            # 7) Get the upscaled image URL
            upscaled_image_data = result["image"]
            upscaled_image_url = upscaled_image_data["url"]

            # 8) Download the upscaled image
            resp = requests.get(upscaled_image_url)
            if resp.status_code != 200:
                raise ConnectionError(f"Failed to download upscaled image. HTTP {resp.status_code}")

            pil_img = Image.open(BytesIO(resp.content))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # 9) Save the image & metadata
            number = self.get_next_number()
            
            generation_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "fal-ai/recraft/upscale/crisp",
                "parameters": {
                    "enable_safety_checker": enable_safety_checker
                },
                "input_image_url": image_url,
                "output_image_info": upscaled_image_data,
                "fal_result": result
            }

            image_path, metadata_path = self.save_image_and_metadata(pil_img, generation_info, number)
            print(f"[Recraft Upscale] Saved upscaled image -> {image_path}")
            print(f"[Recraft Upscale] Saved metadata -> {metadata_path}")

            # 10) Convert to ComfyUI's IMAGE format (1, H, W, 3)
            img_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            img_tensor = img_tensor.unsqueeze(0)

            # 11) Return the image tensor & metadata JSON string
            return (img_tensor, json.dumps(generation_info, indent=2))

        except Exception as e:
            # Return an empty image and an error message in JSON
            error_info = {
                "error": f"Recraft upscale failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "fal-ai/recraft/upscale/crisp"
            }
            print(f"[Recraft Upscale] Error: {str(e)}")
            return (empty_image, json.dumps(error_info, indent=2))

    def tensor_to_tempfile(self, tensor):
        """
        Convert a ComfyUI IMAGE tensor to a PNG file 
        Return an open file in "rb" mode (caller must close/delete).
        """
        pil_img = self.tensor_to_pil(tensor)
        fd, filename = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil_img.save(filename, format="PNG")
        return open(filename, "rb")

    def tensor_to_pil(self, tensor):
        """
        Convert tensor to PIL Image: handle (B, H, W, C) or (C, H, W).
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
        Return NaN means no caching for this node.
        """
        return float("NaN")

    def interrupt(self):
        """
        Interrupt method for consistency.
        """
        print("[Recraft Upscale] Interrupt called.")


# Register with ComfyUI
NODE_CLASS_MAPPINGS = {
    "FalAPI_recraft_upscale": FalAPI_recraft_upscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPI_recraft_upscale": "FAL Recraft Crisp Upscale"
}

__all__ = ["FalAPI_recraft_upscale"] 
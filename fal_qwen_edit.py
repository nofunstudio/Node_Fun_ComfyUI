import os
import time
import requests
import tempfile
import json
import random
import numpy as np
import torch
import asyncio
import folder_paths
import fal_client
from io import BytesIO
from PIL import Image
import base64

class FalAPI_QwenEditPlus:
    @classmethod
    def INPUT_TYPES(cls):
        """
        An image generation node for `fal-ai/qwen-image-edit-plus-lora`.
        """
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("FAL_KEY", ""),
                    "display": "FAL API Key"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful painting",
                    "display": "Prompt"
                }),
                "image1": ("IMAGE", {"display": "Image 1"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "display": "Width"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "display": "Height"}),
            },
            "optional": {
                "image2": ("IMAGE", {"display": "Image 2 (optional)"}),
                "image3": ("IMAGE", {"display": "Image 3 (optional)"}),
                "num_inference_steps": ("INT", {
                    "default": 28,
                    "min": 2,
                    "max": 50,
                    "step": 1,
                    "display": "Steps"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "Guidance Scale (CFG)"
                }),
                "lora_url": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "display": "LoRA URL (optional)"
                }),
                "lora_scale": ("FLOAT", {
                    "default": 0.84,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "LoRA Scale"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "display": "Seed (-1 for random)"
                }),
                "output_format": (["jpeg", "png"], {
                    "default": "png",
                    "display": "Output Format"
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                    "display": "Enable Safety Checker"
                }),
                "acceleration": (["none", "regular"], {
                    "default": "regular",
                    "display": "Acceleration"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("image", "image_path", "generation_info", "generation_time",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL"

    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
        self.output_dir = os.path.join("output", "API", "FAL", "qwen-edit-plus")
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def get_next_number(self):
        valid_exts = {".png", ".jpg"}
        files = [f for f in os.listdir(self.output_dir) 
                 if os.path.splitext(f)[1].lower() in valid_exts]
        numbers = []
        for file_name in files:
            base, _ = os.path.splitext(file_name)
            try:
                numbers.append(int(base))
            except ValueError:
                pass
        return max(numbers) + 1 if numbers else 1

    def save_image_and_metadata(self, pil_img, generation_info, number):
        filename = f"{number:03d}.png"
        filepath = os.path.join(self.output_dir, filename)
        pil_img.save(filepath, format="PNG")

        metadata_filename = f"{number:03d}_metadata.json"
        metadata_filepath = os.path.join(self.metadata_dir, metadata_filename)
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(generation_info, f, indent=4, ensure_ascii=False)
        return filepath, metadata_filepath

    def on_queue_update(self, update):
        try:
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(f"[QwenEditPlus] {log['message']}")
        except Exception as e:
            print(f"[QwenEditPlus] Error in queue update: {e}")

    def tensor_to_tempfile(self, tensor):
        pil_img = self.tensor_to_pil(tensor)
        fd, filename = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil_img.save(filename, format="PNG")
        return filename

    def tensor_to_pil(self, tensor):
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)
        
    def pil_to_tensor(self, pil_img):
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]

    def format_generation_time(self, elapsed_seconds):
        """Format elapsed time as 'seconds-centiseconds' (e.g., '14-50' for 14.50 seconds)"""
        seconds = int(elapsed_seconds)
        centiseconds = int((elapsed_seconds - seconds) * 100)
        return f"{seconds}-{centiseconds:02d}"

    async def generate_image(self, api_key, prompt, image1, width, height,
                             image2=None, image3=None, num_inference_steps=28, guidance_scale=4.0,
                             lora_url="", lora_scale=0.84, seed=-1, output_format="png", 
                             enable_safety_checker=True, acceleration="regular"):
        
        print("[QwenEditPlus] Starting image generation process...")
        start_time = time.time()
        
        if not api_key:
            raise ValueError("A FAL API key is required.")
        
        os.environ["FAL_KEY"] = api_key
        
        temp_files = []
        
        try:
            image_urls = []
            all_images = [image1, image2, image3]
            
            # Upload images
            for i, img_tensor in enumerate(all_images):
                if img_tensor is not None:
                    print(f"[QwenEditPlus] Processing input image {i+1}...")
                    
                    # Convert to temp file and upload
                    temp_path = await asyncio.to_thread(self.tensor_to_tempfile, img_tensor)
                    temp_files.append(temp_path)
                    
                    print(f"[QwenEditPlus] Uploading image {i+1}...")
                    url = await asyncio.to_thread(fal_client.upload_file, temp_path)
                    image_urls.append(url)
            
            if not image_urls:
                raise ValueError("At least one image input is required.")

            # Prepare arguments
            arguments = {
                "prompt": prompt,
                "image_urls": image_urls,
                "image_size": {"width": width, "height": height},
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "enable_safety_checker": enable_safety_checker,
                "output_format": output_format,
                "acceleration": acceleration,
                "sync_mode": False,
                "num_images": 1
            }

            # Add LoRA if provided
            if lora_url.strip():
                arguments["loras"] = [{"path": lora_url, "scale": lora_scale}]
            
            if seed != -1:
                arguments["seed"] = seed

            print(f"[QwenEditPlus] Arguments: {json.dumps(arguments, indent=2)}")
            
            endpoint = "fal-ai/qwen-image-edit-2511/lora"
            print(f"[QwenEditPlus] Submitting to {endpoint}...")

            # Call API
            result = await asyncio.to_thread(
                fal_client.subscribe,
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=self.on_queue_update,
            )

            # Process results
            if not result:
                raise RuntimeError("No result returned from API")
            
            output_images = result.get("images", [])
            if not output_images:
                raise RuntimeError(f"No images in result: {result.keys()}")

            print(f"[QwenEditPlus] Downloaded {len(output_images)} images.")

            final_tensors = []
            saved_paths = []

            for img_info in output_images:
                img_url = img_info.get("url")
                if not img_url: continue
                
                # Download
                resp = await asyncio.to_thread(requests.get, img_url)
                if resp.status_code == 200:
                    pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
                    output_tensor = self.pil_to_tensor(pil_img)
                    final_tensors.append(output_tensor)

                    # Save
                    number = self.get_next_number()
                    gen_info = {
                        "prompt": prompt,
                        "parameters": arguments,
                        "result": result
                    }
                    path, _ = await asyncio.to_thread(self.save_image_and_metadata, pil_img, gen_info, number)
                    saved_paths.append(path)

            if not final_tensors:
                raise RuntimeError("Failed to process output images")
            
            if len(final_tensors) > 1:
                batch_tensor = torch.cat(final_tensors, dim=0)
            else:
                batch_tensor = final_tensors[0]

            generation_time = self.format_generation_time(time.time() - start_time)
            print(f"[QwenEditPlus] Generation completed in {generation_time} seconds")
            return (batch_tensor, ";".join(saved_paths), json.dumps(result, indent=2), generation_time)

        except Exception as e:
            print(f"[QwenEditPlus] Error: {str(e)}")
            # Clean up
            for f in temp_files:
                try: os.remove(f)
                except: pass
                
            error_info = {
                "error": f"QwenEditPlus generation failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            generation_time = self.format_generation_time(time.time() - start_time)
            return (torch.zeros((1, 64, 64, 3)), "", json.dumps(error_info, indent=2), generation_time)
        
        finally:
            # Cleanup temp inputs
            for f in temp_files:
                try: os.remove(f)
                except: pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "FalAPI_QwenEditPlus": FalAPI_QwenEditPlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPI_QwenEditPlus": "FAL Qwen Edit Plus"
}

__all__ = ["FalAPI_QwenEditPlus"]



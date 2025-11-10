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
from io import BytesIO
from PIL import Image
import base64

class WaveSpeedAI_Image:
    @classmethod
    def INPUT_TYPES(cls):
        """
        An image generation node for WavespeedAI's qwen-image/edit-plus-lora API.
        Takes up to 3 images, a prompt, and other parameters to generate a new image.
        """
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("WAVESPEED_API_KEY", ""),
                    "display": "WaveSpeedAI API Key"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful painting",
                    "display": "Prompt"
                }),
                "image1": ("IMAGE", {"display": "Image 1"}),
                "width": ("INT", {"default": 886, "min": 64, "max": 4096, "step": 8, "display": "Width"}),
                "height": ("INT", {"default": 1182, "min": 64, "max": 4096, "step": 8, "display": "Height"}),
            },
            "optional": {
                "image2": ("IMAGE", {"display": "Image 2 (optional)"}),
                "image3": ("IMAGE", {"display": "Image 3 (optional)"}),
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
                "output_format": (["jpeg", "png", "webp"], {
                    "default": "jpeg",
                    "display": "Output Format"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("image", "image_path", "generation_info",)
    FUNCTION = "generate_image"
    CATEGORY = "WaveSpeedAI"

    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
        self.metadata_dir = os.path.join(self.temp_dir, "wavespeed_metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

    def get_temp_filename(self, output_format="jpeg"):
        timestamp = int(time.time() * 1000)
        random_id = random.randint(1000, 9999)
        return f"wavespeed_{timestamp}_{random_id}.{output_format}"

    def save_image_and_metadata(self, image_content, generation_info, filename):
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(image_content)

        metadata_filename = f"{os.path.splitext(filename)[0]}_metadata.json"
        metadata_filepath = os.path.join(self.metadata_dir, metadata_filename)
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(generation_info, f, indent=4, ensure_ascii=False)
        return filepath, metadata_filepath

    async def generate_image(self, api_key, prompt, image1, width, height, 
                             image2=None, image3=None, lora_url="", lora_scale=0.84, 
                             seed=-1, output_format="jpeg"):
        print("[WaveSpeedAI] Starting image generation process...")
        
        if not api_key:
            raise ValueError("A WaveSpeedAI API key is required.")
        if not prompt or not prompt.strip():
            raise ValueError("A prompt is required.")

        try:
            image_urls = []
            all_images = [image1, image2, image3]
            for i, img_tensor in enumerate(all_images):
                if img_tensor is not None:
                    print(f"[WaveSpeedAI] Processing input image {i+1}...")
                    data_uri = await asyncio.to_thread(self.tensor_to_data_uri, img_tensor)
                    image_urls.append(data_uri)
            
            if not image_urls:
                raise ValueError("At least one image input is required.")

            payload = {
                "enable_base64_output": False,
                "enable_sync_mode": False,
                "images": image_urls,
                "output_format": output_format,
                "prompt": prompt,
                "size": f"{width}*{height}"
            }

            if lora_url.strip():
                payload["loras"] = [{"path": lora_url, "scale": lora_scale}]
            
            if seed != -1:
                payload["seed"] = seed

            print("[WaveSpeedAI] Submitting generation task...")
            submit_url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/qwen-image/edit-plus-lora"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            
            response = await asyncio.to_thread(requests.post, submit_url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                raise ConnectionError(f"Failed to submit task. HTTP {response.status_code}: {response.text}")
            
            request_id = response.json()["data"]["id"]
            print(f"[WaveSpeedAI] Task submitted successfully. Request ID: {request_id}")

            result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
            
            output_url = ""
            final_result = {}
            while True:
                print("[WaveSpeedAI] Polling for result...")
                response = await asyncio.to_thread(requests.get, result_url, headers={"Authorization": f"Bearer {api_key}"})
                if response.status_code == 200:
                    result = response.json()["data"]
                    status = result["status"]
                    if status == "completed":
                        print("[WaveSpeedAI] Task completed.")
                        output_url = result["outputs"][0]
                        final_result = result
                        break
                    elif status == "failed":
                        raise RuntimeError(f"Task failed: {result.get('error', 'Unknown error')}")
                    else:
                        print(f"[WaveSpeedAI] Status: {status}. Waiting...")
                else:
                    raise ConnectionError(f"Failed to poll for results. HTTP {response.status_code}: {response.text}")
                await asyncio.sleep(2)

            print(f"[WaveSpeedAI] Downloading image from: {output_url}")
            resp = await asyncio.to_thread(requests.get, output_url)
            if resp.status_code != 200:
                raise ConnectionError(f"Failed to download image. HTTP {resp.status_code}")

            generation_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "qwen-image/edit-plus-lora",
                "parameters": {k: v for k, v in payload.items() if k != 'images'},
                "input_images_count": len(image_urls),
                "wavespeed_result": final_result
            }

            temp_filename = self.get_temp_filename(output_format)
            image_path, metadata_path = await asyncio.to_thread(
                self.save_image_and_metadata, resp.content, generation_info, temp_filename
            )
            print(f"[WaveSpeedAI] Saved temp image -> {image_path}")
            print(f"[WaveSpeedAI] Saved metadata -> {metadata_path}")
            
            pil_image = Image.open(BytesIO(resp.content))
            output_tensor = self.pil_to_tensor(pil_image)
            
            return (output_tensor, image_path, json.dumps(generation_info, indent=2))

        except Exception as e:
            error_info = {
                "error": f"WaveSpeedAI generation failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"[WaveSpeedAI] Error: {str(e)}")
            return (None, "", json.dumps(error_info, indent=2))

    def tensor_to_data_uri(self, tensor, image_format="PNG"):
        pil_img = self.tensor_to_pil(tensor)
        buffered = BytesIO()
        pil_img.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{image_format.lower()};base64,{img_str}"

    def tensor_to_pil(self, tensor):
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)
        
    def pil_to_tensor(self, pil_img):
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

# Register with ComfyUI
NODE_CLASS_MAPPINGS = {
    "WaveSpeedAI_Image": WaveSpeedAI_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveSpeedAI_Image": "WaveSpeedAI Image Generation"
}

__all__ = ["WaveSpeedAI_Image"]

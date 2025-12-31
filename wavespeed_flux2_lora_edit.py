import os
import time
import requests
import json
import random
import numpy as np
import torch
import asyncio
import folder_paths
from io import BytesIO
from PIL import Image
import base64

class WaveSpeedAI_Flux2LoraEdit:
    @classmethod
    def INPUT_TYPES(cls):
        """
        An image generation node for `wavespeed-ai/flux-2-dev/edit-lora`.
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
                    "default": "make this picture berthe_morisot style",
                    "display": "Prompt"
                }),
                "image1": ("IMAGE", {"display": "Image 1"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8, "display": "Width"}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8, "display": "Height"}),
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
                    "default": 1.0,
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

    def get_temp_filename(self, output_format="png"):
        timestamp = int(time.time() * 1000)
        random_id = random.randint(1000, 9999)
        return f"wavespeed_flux2_{timestamp}_{random_id}.{output_format}"

    def save_image_and_metadata(self, image_content, generation_info, filename):
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(image_content)

        metadata_filename = f"{os.path.splitext(filename)[0]}_metadata.json"
        metadata_filepath = os.path.join(self.metadata_dir, metadata_filename)
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(generation_info, f, indent=4, ensure_ascii=False)
        return filepath, metadata_filepath

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

    async def generate_image(self, api_key, prompt, image1, width, height, image2=None, image3=None, 
                             lora_url="", lora_scale=1.0, seed=-1):
        
        print("[WaveSpeedAI Flux2] Starting image generation process...")
        
        if not api_key:
            raise ValueError("A WaveSpeedAI API key is required.")
        
        os.environ["WAVESPEED_API_KEY"] = api_key
        
        try:
            image_urls = []
            all_images = [image1, image2, image3]
            
            # Convert images to Data URIs
            for i, img_tensor in enumerate(all_images):
                if img_tensor is not None:
                    print(f"[WaveSpeedAI Flux2] Processing input image {i+1}...")
                    data_uri = await asyncio.to_thread(self.tensor_to_data_uri, img_tensor)
                    image_urls.append(data_uri)
            
            if not image_urls:
                raise ValueError("At least one image input is required.")

            # Prepare payload
            payload = {
                "enable_base64_output": False,
                "enable_sync_mode": False,
                "images": image_urls,
                "prompt": prompt,
                "seed": seed,
                "size": f"{width}*{height}"
            }

            # Add LoRA if provided
            if lora_url.strip():
                payload["loras"] = [{"path": lora_url, "scale": lora_scale}]
            else:
                payload["loras"] = [] 

            print(f"[WaveSpeedAI Flux2] Payload: {json.dumps({k:v for k,v in payload.items() if k!='images'}, indent=2)}")
            
            url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/flux-2-dev/edit-lora"
            headers = {
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {api_key}"
            }

            print(f"[WaveSpeedAI Flux2] Submitting to {url}...")
            
            # Submit task
            response = await asyncio.to_thread(requests.post, url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                raise ConnectionError(f"Failed to submit task. HTTP {response.status_code}: {response.text}")
            
            response_data = response.json()
            if "data" in response_data:
                request_id = response_data["data"]["id"]
            else:
                # Handle cases where 'data' might be missing or different structure if any
                raise ValueError(f"Unexpected response structure: {response_data}")

            print(f"[WaveSpeedAI Flux2] Task submitted successfully. Request ID: {request_id}")

            # Poll for results
            result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
            
            output_url = ""
            final_result = {}
            
            while True:
                response = await asyncio.to_thread(requests.get, result_url, headers=headers)
                if response.status_code == 200:
                    result = response.json()["data"]
                    status = result["status"]
                    
                    if status == "completed":
                        print("[WaveSpeedAI Flux2] Task completed.")
                        output_url = result["outputs"][0]
                        final_result = result
                        break
                    elif status == "failed":
                        raise RuntimeError(f"Task failed: {result.get('error', 'Unknown error')}")
                    else:
                        print(f"[WaveSpeedAI Flux2] Status: {status}. Waiting...")
                else:
                    print(f"[WaveSpeedAI Flux2] Polling error: {response.status_code}")
                    if 400 <= response.status_code < 500:
                         raise ConnectionError(f"Polling failed with client error {response.status_code}: {response.text}")
                
                await asyncio.sleep(2) # Poll every 2 seconds

            # Download result
            print(f"[WaveSpeedAI Flux2] Downloading image from: {output_url}")
            resp = await asyncio.to_thread(requests.get, output_url)
            if resp.status_code != 200:
                raise ConnectionError(f"Failed to download image. HTTP {resp.status_code}")

            # Save and return
            generation_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "flux-2-dev/edit-lora",
                "parameters": {k: v for k, v in payload.items() if k != 'images'},
                "input_images_count": len(image_urls),
                "wavespeed_result": final_result
            }

            temp_filename = self.get_temp_filename()
            image_path, metadata_path = await asyncio.to_thread(
                self.save_image_and_metadata, resp.content, generation_info, temp_filename
            )
            
            pil_image = Image.open(BytesIO(resp.content)).convert("RGB")
            output_tensor = self.pil_to_tensor(pil_image)
            
            return (output_tensor, image_path, json.dumps(generation_info, indent=2))

        except Exception as e:
            print(f"[WaveSpeedAI Flux2] Error: {str(e)}")
            error_info = {
                "error": f"WaveSpeedAI Flux2 generation failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            return (torch.zeros((1, 64, 64, 3)), "", json.dumps(error_info, indent=2))

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "WaveSpeedAI_Flux2LoraEdit": WaveSpeedAI_Flux2LoraEdit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveSpeedAI_Flux2LoraEdit": "WaveSpeedAI Flux 2 LoRA Edit"
}

__all__ = ["WaveSpeedAI_Flux2LoraEdit"]

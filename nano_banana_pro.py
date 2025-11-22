import os
import time
import requests
import tempfile
import json
import numpy as np
import torch
import asyncio
import fal_client
from io import BytesIO
from PIL import Image
import folder_paths

class FalAPI_NanoBananaPro:
    @classmethod
    def INPUT_TYPES(cls):
        """
        An image generation node for Nano Banana Pro (via FAL).
        Takes up to 4 images, a prompt, and other parameters.
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
                    "default": "A beautiful masterpiece",
                    "display": "Prompt"
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "display": "Num Images"
                }),
                "image1": ("IMAGE", {"display": "Image 1"}),
                "aspect_ratio": ([
                    "custom",
                    "16:9",
                    "4:3",
                    "1:1",
                    "3:4",
                    "9:16",
                    "21:9"
                ], {
                    "default": "1:1",
                    "display": "Aspect Ratio"
                }),
                "resolution": ([
                    "1K",
                    "2K",
                    "4K"
                ], {
                    "default": "1K",
                    "display": "Resolution"
                }),
            },
            "optional": {
                "image2": ("IMAGE", {"display": "Image 2 (optional)"}),
                "image3": ("IMAGE", {"display": "Image 3 (optional)"}),
                "image4": ("IMAGE", {"display": "Image 4 (optional)"}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "display": "Seed (-1 for random)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("images", "image_paths", "generation_info",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL"

    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
        self.output_dir = os.path.join("output", "API", "FAL", "nano-banana-pro")
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
                    print(f"[NanoBananaPro] {log['message']}")
        except Exception as e:
            print(f"[NanoBananaPro] Error in queue update: {e}")

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

    async def generate_image(self, api_key, prompt, num_images, image1, aspect_ratio, resolution,
                       image2=None, image3=None, image4=None, seed=-1):
        print("[NanoBananaPro] Starting generation process...")

        if not api_key:
            raise ValueError("A FAL API Key is required.")
        
        os.environ["FAL_KEY"] = api_key

        # Prepare inputs
        input_images = [image1, image2, image3, image4]
        image_urls = []
        temp_files = []

        try:
            # Upload images
            for i, img in enumerate(input_images):
                if img is not None:
                    print(f"[NanoBananaPro] Uploading image {i+1}...")
                    # Run file conversion and upload in thread pool
                    temp_path = await asyncio.to_thread(self.tensor_to_tempfile, img)
                    temp_files.append(temp_path)
                    
                    url = await asyncio.to_thread(fal_client.upload_file, temp_path)
                    image_urls.append(url)

            if not image_urls:
                raise ValueError("At least one image input is required.")

            # Prepare arguments
            arguments = {
                "prompt": prompt,
                "num_images": num_images,
                "image_urls": image_urls,
                "resolution": resolution,
                "output_format": "png",
                "sync_mode": False
            }

            # Handle aspect ratio or custom resolution
            if aspect_ratio != "custom":
                 arguments["aspect_ratio"] = aspect_ratio

            # Seed handling
            if seed != -1:
                arguments["seed"] = seed

            print(f"[NanoBananaPro] Arguments: {json.dumps(arguments, indent=2)}")

            # Manual submission to get request_id for polling control
            submit_url = "https://queue.fal.run/fal-ai/nano-banana-pro/edit"
            headers = {
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json"
            }
            
            print(f"[NanoBananaPro] Submitting job to {submit_url}...")
            
            # Submit job
            response = await asyncio.to_thread(requests.post, submit_url, headers=headers, json=arguments)
            
            if response.status_code != 200:
                 raise RuntimeError(f"Submission failed: {response.status_code} {response.text}")
                 
            data = response.json()
            request_id = data.get("request_id")
            if not request_id:
                raise RuntimeError(f"No request_id in response: {data}")
                
            print(f"[NanoBananaPro] Request ID: {request_id}")
            
            status_url = f"https://queue.fal.run/fal-ai/nano-banana-pro/requests/{request_id}/status?logs=true"
            result_url = f"https://queue.fal.run/fal-ai/nano-banana-pro/requests/{request_id}"
            
            start_time = time.time()
            TIMEOUT = 75
            
            result = None
            
            # Manual polling loop
            while True:
                if time.time() - start_time > TIMEOUT:
                    print("[NanoBananaPro] API request timed out after 75 seconds. Returning black image.")
                    
                    # Stop polling, return black image
                    size_map = {"1K": 1024, "2K": 2048, "4K": 4096}
                    dim = size_map.get(resolution, 1024)
                    
                    batch_size = num_images
                    black_tensor = torch.zeros((batch_size, dim, dim, 3))
                    
                    timeout_info = {
                        "error": "Timeout",
                        "message": "API request timed out after 75 seconds",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "prompt": prompt
                    }
                    
                    black_pil = Image.new("RGB", (dim, dim), (0, 0, 0))
                    
                    saved_paths = []
                    for _ in range(batch_size):
                        number = self.get_next_number()
                        path, _ = await asyncio.to_thread(self.save_image_and_metadata, black_pil, timeout_info, number)
                        saved_paths.append(path)
                    
                    return (black_tensor, ";".join(saved_paths), json.dumps(timeout_info, indent=2))
                
                # Check status
                try:
                    status_resp = await asyncio.to_thread(requests.get, status_url, headers=headers)
                    if status_resp.status_code == 200:
                        status_data = status_resp.json()
                        status = status_data.get("status")
                        
                        # Print logs if available
                        logs = status_data.get("logs", [])
                        if logs:
                            for log in logs:
                                msg = log.get("message", "")
                                if msg: print(f"[NanoBananaPro] {msg}")

                        if status == "COMPLETED":
                            # Fetch result
                            result_resp = await asyncio.to_thread(requests.get, result_url, headers=headers)
                            result = result_resp.json()
                            break
                        elif status == "FAILED":
                            error_msg = status_data.get("error", "Unknown error")
                            raise RuntimeError(f"Job failed: {error_msg}")
                except requests.exceptions.RequestException as e:
                    print(f"[NanoBananaPro] Network error checking status: {e}")
                    # Continue polling loop despite network error, unless timeout is reached
                    await asyncio.sleep(1)
                    continue
                
                await asyncio.sleep(1)

            # Process results
            if not result:
                raise RuntimeError("No result returned from API")
            
            output_images = []
            if "images" in result:
                output_images = result["images"]
            elif "image" in result:
                output_images = [result["image"]]
            
            if not output_images:
                 raise RuntimeError(f"No images in result: {result.keys()}")

            print(f"[NanoBananaPro] Downloaded {len(output_images)} images.")

            final_tensors = []
            saved_paths = []

            for img_info in output_images:
                img_url = img_info.get("url")
                if not img_url: continue
                
                # Run download in thread pool
                resp = await asyncio.to_thread(requests.get, img_url)
                
                if resp.status_code == 200:
                    pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
                    
                    # Convert to tensor
                    img_array = np.array(pil_img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    final_tensors.append(img_tensor)

                    # Save
                    number = self.get_next_number()
                    gen_info = {
                        "prompt": prompt,
                        "parameters": arguments,
                        "result": result
                    }
                    # Run save in thread pool
                    path, _ = await asyncio.to_thread(self.save_image_and_metadata, pil_img, gen_info, number)
                    saved_paths.append(path)

            if not final_tensors:
                raise RuntimeError("Failed to process output images")

            # Stack tensors if multiple
            if len(final_tensors) > 1:
                # Ensure same size before stacking?
                # Usually yes. If not, we might fail or need resizing.
                # For now assume API returns consistent sizes.
                batch_tensor = torch.cat(final_tensors, dim=0)
            else:
                batch_tensor = final_tensors[0]

            return (batch_tensor, ";".join(saved_paths), json.dumps(result, indent=2))

        except Exception as e:
            print(f"[NanoBananaPro] Error: {e}")
            # Cleanup
            for f in temp_files:
                try: os.remove(f)
                except: pass
            
            # Return empty/error
            empty = torch.zeros((1, 64, 64, 3))
            return (empty, "", str(e))
        
        finally:
             # Cleanup temp inputs
            for f in temp_files:
                try: os.remove(f)
                except: pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "FalAPI_NanoBananaPro": FalAPI_NanoBananaPro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPI_NanoBananaPro": "Nano Banana Pro (FAL)"
}

__all__ = ["FalAPI_NanoBananaPro"]

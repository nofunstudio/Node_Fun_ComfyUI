import os
import time
import requests
import tempfile
from PIL import Image
import numpy as np
import torch
import replicate
from io import BytesIO
import json
import threading
import asyncio
import io

class ReplicateAPI_flux_1_1_pro_ultra:
    @classmethod
    def INPUT_TYPES(cls):
        """
        We add an OPTIONAL input 'image_prompt' of type IMAGE,
        so the user can connect a Load Image node.
        """
        return {
            "required": {
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "display": "Replicate API Token"
                }),
                "model": ([
                    "black-forest-labs/flux-1.1-pro-ultra",
                    # add any other Replicate models you want here...
                ], {
                    "default": "black-forest-labs/flux-1.1-pro-ultra"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A blackhole in space"
                }),
                "image_prompt_strength": ("INT", {
                    "default": .1,
                    "min": .1,
                    "max": 1,
                    "step": .01
                }),
                "number_of_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
                "timeout": ("INT", {
                    "default": 300,
                    "min": 60,
                    "max": 1800,
                    "step": 60,
                    "display": "Timeout (seconds)"
                }),
            },
            "optional": {
                # This allows an optional image input
                "image_prompt": ("IMAGE", {})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "generation_info",)
    FUNCTION = "generate"
    CATEGORY = "Replicate"

    def __init__(self):
        self.output_dir = "output/API/Replicate"
        self.metadata_dir = "output/API/Replicate/metadata"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        self._interrupt_event = threading.Event()

    def get_next_number(self):
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        if not files:
            return 1
        numbers = [int(f.split('.')[0]) for f in files]
        return max(numbers) + 1

    def create_filename(self, number):
        return f"{number:03d}.png"

    def save_image_and_metadata(self, img, generation_info, number):
        filename = self.create_filename(number)
        filepath = os.path.join(self.output_dir, filename)
        
        # Save image
        img.save(filepath)

        # Create metadata filename
        metadata_filename = f"{number:03d}_metadata.json"
        metadata_filepath = os.path.join(self.metadata_dir, metadata_filename)
        
        # Save metadata
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(generation_info, f, indent=4, ensure_ascii=False)

        return filepath, metadata_filepath

    async def generate_single_image_async(self, input_data, api_token, model, image_prompt_tensor=None):
        """
        A single async "job" that calls replicate.run() in a worker thread.
        If 'image_prompt_tensor' is provided, convert it to a local file
        and pass it as 'image_prompt'.
        """
        try:
            # Set the Replicate API token
            os.environ["REPLICATE_API_TOKEN"] = api_token

            # If we have an image prompt from ComfyUI, turn it into a file
            image_prompt_file = None
            if image_prompt_tensor is not None:
                # Convert the incoming torch tensor to PIL
                pil_image = self.tensor_to_pil(image_prompt_tensor)

                # Write to a temporary file
                tmp_fd, tmp_filename = tempfile.mkstemp(suffix=".png")
                os.close(tmp_fd)  # We'll reopen as "rb" below.
                pil_image.save(tmp_filename, format="PNG")
                
                # We'll open the file for reading as replicate expects a file object
                image_prompt_file = open(tmp_filename, "rb")
                # Provide it in the input dictionary
                input_data["image_prompt"] = image_prompt_file

            # Actually call replicate.run(...) in a thread to avoid blocking
            def replicate_run_wrapper():
                return replicate.run(model, input=input_data)

            output = await asyncio.to_thread(replicate_run_wrapper)

            # Clean up the image prompt file if we used one
            if image_prompt_file is not None:
                image_prompt_file.close()
                os.remove(image_prompt_file.name)

            if not output:
                raise ValueError("No valid result received from replicate.run().")

            # Some Replicate models return a list of URLs, or a single URL/string
            if isinstance(output, list):
                image_url = output[0]
            else:
                image_url = output

            # Download the resulting image
            image_response = requests.get(image_url)
            if image_response.status_code != 200:
                raise ConnectionError(f"Failed to download image: Status code {image_response.status_code}")

            # Convert into PIL
            img = Image.open(BytesIO(image_response.content)).convert("RGB")

            # --- Remove non-serializable file object before saving metadata ---
            safe_input_data = dict(input_data)
            # pop() ensures we remove the open file or anything else that won't serialize
            safe_input_data.pop("image_prompt", None)

            # Save the image and metadata
            number = self.get_next_number()
            generation_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": safe_input_data,
                "replicate_output": str(output)
            }
            # ---

            image_path, metadata_path = self.save_image_and_metadata(img, generation_info, number)
            print(f"Saved image to: {image_path}")
            print(f"Saved metadata to: {metadata_path}")

            # Convert to a torch tensor (Batch x Height x Width x Channels)
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

            return img_tensor, generation_info

        except Exception as e:
            print(f"Generation error: {str(e)}")
            raise Exception(f"Error generating image: {str(e)}")

    def generate(self, api_token, model, prompt, number_of_images=1, seed=-1, timeout=300, image_prompt=None, image_prompt_strength=.1):
        """
        The main entry point for ComfyUI. Gathers multiple images if requested,
        passes a (possibly optional) image_prompt, and returns combined results.
        """
        if not api_token:
            raise ValueError("API token is required.")

        self._interrupt_event.clear()
        empty_image = torch.zeros((1, 1024, 1024, 3))

        try:
            images = []
            infos = []
            failed_jobs = []

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def process_all_images():
                tasks = []
                for i in range(number_of_images):
                    if self._interrupt_event.is_set():
                        break

                    # Build input data
                    input_data = {
                        "prompt": prompt,
                        "raw": False,
                        "output_format": "png",
                    }

                    # Seed logic
                    if seed != -1:
                        current_seed = seed + i
                    else:
                        current_seed = np.random.randint(0, 2147483647)

                    input_data["seed"] = current_seed

                    # Add more replicate model parameters if needed here
                    # e.g., "aspect_ratio", "safety_tolerance", etc.

                    tasks.append(
                        self.generate_single_image_async(
                            input_data,
                            api_token,
                            model,
                            image_prompt_tensor=image_prompt, 
                            image_prompt_strength=image_prompt_strength
                        )
                    )
                
                return await asyncio.gather(*tasks, return_exceptions=True)

            try:
                results = loop.run_until_complete(process_all_images())
            finally:
                loop.close()

            # Process the results
            for result in results:
                if isinstance(result, Exception):
                    failed_jobs.append({'error': str(result)})
                else:
                    img_tensor, generation_info = result
                    images.append(img_tensor)
                    infos.append(generation_info)

            # If no images were successful
            if not images:
                generation_info = {
                    "error": "All generation jobs failed",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "failed_jobs": failed_jobs
                }
                return (empty_image, json.dumps(generation_info, indent=2))

            # Combine all images into a batch
            combined_tensor = torch.cat(images, dim=0)

            combined_info = {
                "successful_generations": len(images),
                "total_requested": number_of_images,
                "generation_parameters": {
                    "prompt": prompt,
                    "initial_seed": seed,
                    "image_prompt_strength": image_prompt_strength
                },
                "individual_results": infos,
                "failed_jobs": failed_jobs if failed_jobs else None
            }

            return (combined_tensor, json.dumps(combined_info, indent=2))

        except Exception as e:
            generation_info = {
                "error": f"Replicate generation failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            return (empty_image, json.dumps(generation_info, indent=2))

    def tensor_to_pil(self, tensor):
        """
        Helper method: Convert a ComfyUI image tensor (CxHxW or HxWxC) to a PIL Image.
        Typically, ComfyUI image is (B, H, W, C) or (H, W, C). Let's handle the common case.
        """
        # If the tensor is 4D, remove batch dimension
        if len(tensor.shape) == 4:
            # shape is (B, H, W, C), typically B=1
            tensor = tensor[0]

        # Now shape should be (H, W, C)
        arr = tensor.cpu().numpy()
        arr = (arr * 255).clip(0, 255).astype("uint8")
        pil_image = Image.fromarray(arr)
        return pil_image

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def interrupt(self):
        print("Interrupting Replicate generation...")
        self._interrupt_event.set()

NODE_CLASS_MAPPINGS = {
    "Replicate flux 1.1 pro ultra": ReplicateAPI_flux_1_1_pro_ultra
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplicateAPI_flux_1_1_pro_ultra": "Replicate flux 1.1 pro ultra"
}

__all__ = ["ReplicateAPI_flux_1_1_pro_ultra"]

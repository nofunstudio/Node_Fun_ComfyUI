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

class FalAPI_kling_video:
    @classmethod
    def INPUT_TYPES(cls):
        """
        A video generation node for `fal-ai/kling-video/v2.1/standard/image-to-video`.
        Takes an image input, prompt, and generates a video using Kling Video API.
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
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic video transformation",
                    "display": "Video Prompt"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("video_path", "generation_info", "generation_time",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL"

    def __init__(self):
        # Directory structure for saving generated videos
        self.output_dir = "output/API/FAL/kling-video"
        self.metadata_dir = os.path.join(self.output_dir, "metadata")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def get_next_number(self):
        """
        Looks at existing .mp4/.mov files and picks the next integer
        filename index (e.g. 001, 002, etc.).
        """
        valid_exts = {".mp4", ".mov", ".avi"}
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
        Zero-padded filenames ending in .mp4 (e.g., '001.mp4').
        """
        return f"{number:03d}.mp4"

    def save_video_and_metadata(self, video_content, generation_info, number):
        """
        Saves the generated video as .mp4, plus metadata as a .json.
        """
        filename = self.create_filename(number)
        filepath = os.path.join(self.output_dir, filename)

        # Save video content
        with open(filepath, 'wb') as f:
            f.write(video_content)

        # Create metadata filename (001_metadata.json)
        metadata_filename = f"{number:03d}_metadata.json"
        metadata_filepath = os.path.join(self.metadata_dir, metadata_filename)

        # Write out metadata
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(generation_info, f, indent=4, ensure_ascii=False)

        return filepath, metadata_filepath

    def on_queue_update(self, update):
        """Handle queue updates and log messages"""
        try:
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(f"[Kling Video] {log['message']}")
        except Exception as e:
            print(f"[Kling Video] Error in queue update: {e}")

    def format_generation_time(self, elapsed_seconds):
        """Format elapsed time as 'seconds-centiseconds' (e.g., '14-50' for 14.50 seconds)"""
        seconds = int(elapsed_seconds)
        centiseconds = int((elapsed_seconds - seconds) * 100)
        return f"{seconds}-{centiseconds:02d}"

    def generate_video(self, api_token, image, prompt):
        """
        Generate video using fal-ai/kling-video/v2.1/standard/image-to-video
        """
        print(f"[Kling Video] Starting video generation process...")
        start_time = time.time()
        
        # For errors, return empty strings
        empty_path = ""
        
        # Make sure we have an API token
        if not api_token:
            error_msg = "A FAL API token is required."
            print(f"[Kling Video] Error: {error_msg}")
            raise ValueError(error_msg)
        
        # Validate inputs
        if image is None:
            error_msg = "Image input is required."
            print(f"[Kling Video] Error: {error_msg}")
            raise ValueError(error_msg)
            
        if not prompt or prompt.strip() == "":
            error_msg = "Prompt input is required."
            print(f"[Kling Video] Error: {error_msg}")
            raise ValueError(error_msg)
            
        print(f"[Kling Video] Input image shape: {image.shape}")
        print(f"[Kling Video] Prompt: {prompt}")

        try:
            # 1) Set up FAL client with API token
            os.environ["FAL_KEY"] = api_token
            print(f"[Kling Video] API token set, processing image...")
            
            # Check if fal_client is working
            try:
                print(f"[Kling Video] Testing fal_client import: {fal_client.__version__ if hasattr(fal_client, '__version__') else 'version unknown'}")
            except Exception as e:
                print(f"[Kling Video] fal_client check failed: {str(e)}")

            # 2) Convert ComfyUI image tensor to temporary file
            try:
                temp_file_path = self.tensor_to_tempfile(image)
                print(f"[Kling Video] Created temp file: {temp_file_path}")
            except Exception as e:
                print(f"[Kling Video] Error creating temp file: {str(e)}")
                raise
            
            # 3) Upload image to FAL storage and get URL
            try:
                print(f"[Kling Video] Uploading image to FAL storage...")
                image_url = fal_client.upload_file(temp_file_path)
                print(f"[Kling Video] Image uploaded successfully: {image_url}")
            except Exception as e:
                print(f"[Kling Video] Error uploading file: {str(e)}")
                # Clean up file before re-raising
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                raise

            # 4) Build the input arguments
            arguments = {
                "prompt": prompt,
                "image_url": image_url
            }

            print(f"[Kling Video] Starting video generation...")

            # 5) Call fal_client.subscribe() 
            try:
                result = fal_client.subscribe(
                    "fal-ai/kling-video/v2.1/standard/image-to-video",
                    arguments=arguments,
                    with_logs=True,
                    on_queue_update=self.on_queue_update,
                )
                print(f"[Kling Video] Video generation completed successfully")
            except Exception as e:
                print(f"[Kling Video] Error during video generation: {str(e)}")
                # Clean up file before re-raising
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                raise

            # 6) Clean up temporary file
            os.remove(temp_file_path)
            print(f"[Kling Video] Cleaned up temp file")

            if not result or not result.get("video"):
                raise ValueError("No valid video result from fal_client.subscribe().")

            # 7) Get the generated video URL
            video_data = result["video"]
            video_url = video_data["url"]

            # 8) Download the generated video
            print(f"[Kling Video] Downloading video from: {video_url}")
            resp = requests.get(video_url)
            if resp.status_code != 200:
                raise ConnectionError(f"Failed to download video. HTTP {resp.status_code}")

            # 9) Save the video & metadata
            number = self.get_next_number()
            
            generation_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "fal-ai/kling-video/v2.1/standard/image-to-video",
                "parameters": {
                    "prompt": prompt
                },
                "input_image_url": image_url,
                "output_video_info": video_data,
                "fal_result": result
            }

            video_path, metadata_path = self.save_video_and_metadata(resp.content, generation_info, number)
            print(f"[Kling Video] Saved video -> {video_path}")
            print(f"[Kling Video] Saved metadata -> {metadata_path}")

            # 10) Return the video path & metadata JSON string
            generation_time = self.format_generation_time(time.time() - start_time)
            print(f"[Kling Video] Generation completed in {generation_time} seconds")
            return (video_path, json.dumps(generation_info, indent=2), generation_time)

        except Exception as e:
            # Return empty path and an error message in JSON
            error_info = {
                "error": f"Kling video generation failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "fal-ai/kling-video/v2.1/standard/image-to-video"
            }
            print(f"[Kling Video] Error: {str(e)}")
            generation_time = self.format_generation_time(time.time() - start_time)
            return (empty_path, json.dumps(error_info, indent=2), generation_time)

    def tensor_to_tempfile(self, tensor):
        """
        Convert a ComfyUI IMAGE tensor to a PNG file 
        Return the file path (caller must delete).
        """
        pil_img = self.tensor_to_pil(tensor)
        fd, filename = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil_img.save(filename, format="PNG")
        return filename

    def tensor_to_pil(self, tensor):
        """
        Convert tensor to PIL Image: handle (B, H, W, C) or (C, H, W).
        """
        print(f"[Kling Video] Converting tensor to PIL - input shape: {tensor.shape}")
        
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # remove batch dimension
            print(f"[Kling Video] Removed batch dimension, new shape: {tensor.shape}")

        arr = tensor.cpu().numpy()
        print(f"[Kling Video] Converted to numpy array, shape: {arr.shape}")
        
        # If shape is (C, H, W), transpose it
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1, 2, 0))
            print(f"[Kling Video] Transposed array, new shape: {arr.shape}")

        arr = (arr * 255).clip(0, 255).astype("uint8")
        print(f"[Kling Video] Converted to uint8, shape: {arr.shape}")
        
        pil_img = Image.fromarray(arr)
        print(f"[Kling Video] Created PIL image: {pil_img.size} mode: {pil_img.mode}")
        return pil_img

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
        print("[Kling Video] Interrupt called.")


# Register with ComfyUI
NODE_CLASS_MAPPINGS = {
    "FalAPI_kling_video": FalAPI_kling_video
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPI_kling_video": "FAL Kling Video Generation"
}

__all__ = ["FalAPI_kling_video"] 
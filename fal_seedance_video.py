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
import fal_client
from comfy_api.input_impl import VideoFromFile

class FalAPI_seedance_video:
    @classmethod
    def INPUT_TYPES(cls):
        """
        A video generation node for `fal-ai/bytedance/seedance/v1.5/pro/image-to-video`.
        Takes an image input, prompt, and generates a video using Seedance Video API.
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
                "aspect_ratio": ([
                    "21:9",
                    "16:9",
                    "4:3",
                    "1:1",
                    "3:4",
                    "9:16",
                    "auto"
                ], {
                    "default": "3:4",
                    "display": "Aspect Ratio"
                }),
                "resolution": ([
                    "480p",
                    "720p",
                    "1080p"
                ], {
                    "default": "1080p",
                    "display": "Resolution"
                }),
                "duration": ([
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "11",
                    "12"
                ], {
                    "default": "5",
                    "display": "Duration (seconds)"
                }),
                "camera_fixed": ("BOOLEAN", {
                    "default": True,
                    "display": "Camera Fixed"
                }),
                "generate_audio": ("BOOLEAN", {
                    "default": True,
                    "display": "Generate Audio"
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                    "display": "Enable Safety Checker"
                }),
            },
            "optional": {
                "end_image": ("IMAGE", {"display": "End Image (optional)"}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "display": "Seed (-1 for random)"
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING",)
    RETURN_NAMES = ("video", "video_path", "generation_info",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL"

    def __init__(self):
        # Use ComfyUI's temp directory for temporary video storage
        self.temp_dir = folder_paths.get_temp_directory()
        self.metadata_dir = os.path.join(self.temp_dir, "seedance_metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

    def get_temp_filename(self):
        """
        Generate a unique temporary filename using timestamp and random component.
        """
        timestamp = int(time.time() * 1000)
        random_id = random.randint(1000, 9999)
        return f"seedance_{timestamp}_{random_id}.mp4"

    def save_video_and_metadata(self, video_content, generation_info, filename):
        """
        Saves the generated video to temp directory, plus metadata as a .json.
        """
        filepath = os.path.join(self.temp_dir, filename)

        # Save video content to temp directory
        with open(filepath, 'wb') as f:
            f.write(video_content)

        # Create metadata filename based on video filename
        metadata_filename = f"{os.path.splitext(filename)[0]}_metadata.json"
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
                    print(f"[Seedance Video] {log['message']}")
        except Exception as e:
            print(f"[Seedance Video] Error in queue update: {e}")

    async def generate_video(self, api_token, image, prompt, aspect_ratio, resolution, 
                      duration, camera_fixed, generate_audio, enable_safety_checker, end_image=None, seed=-1):
        """
        Generate video using fal-ai/bytedance/seedance/v1.5/pro/image-to-video
        (Async execution for parallel processing)
        """
        print(f"[Seedance Video] Starting video generation process...")
        
        # For errors, return empty strings
        empty_path = ""
        
        # Make sure we have an API token
        if not api_token:
            error_msg = "A FAL API token is required."
            print(f"[Seedance Video] Error: {error_msg}")
            raise ValueError(error_msg)
        
        # Validate inputs
        if image is None:
            error_msg = "Image input is required."
            print(f"[Seedance Video] Error: {error_msg}")
            raise ValueError(error_msg)
            
        if not prompt or prompt.strip() == "":
            error_msg = "Prompt input is required."
            print(f"[Seedance Video] Error: {error_msg}")
            raise ValueError(error_msg)
            
        print(f"[Seedance Video] Input image shape: {image.shape}")
        print(f"[Seedance Video] Prompt: {prompt}")
        print(f"[Seedance Video] Settings: aspect_ratio={aspect_ratio}, resolution={resolution}, duration={duration}s")
        print(f"[Seedance Video] Camera fixed: {camera_fixed}, Generate audio: {generate_audio}, Safety checker: {enable_safety_checker}")

        try:
            # 1) Set up FAL client with API token
            os.environ["FAL_KEY"] = api_token
            print(f"[Seedance Video] API token set, processing image...")
            
            # Check if fal_client is working
            try:
                print(f"[Seedance Video] Testing fal_client import: {fal_client.__version__ if hasattr(fal_client, '__version__') else 'version unknown'}")
            except Exception as e:
                print(f"[Seedance Video] fal_client check failed: {str(e)}")

            # 2) Convert ComfyUI image tensor to temporary file (run in thread pool)
            try:
                temp_file_path = await asyncio.to_thread(self.tensor_to_tempfile, image)
                print(f"[Seedance Video] Created temp file: {temp_file_path}")
                
                # Handle end_image if provided
                end_image_temp_path = None
                if end_image is not None:
                    end_image_temp_path = await asyncio.to_thread(self.tensor_to_tempfile, end_image)
                    print(f"[Seedance Video] Created end_image temp file: {end_image_temp_path}")
            except Exception as e:
                print(f"[Seedance Video] Error creating temp file: {str(e)}")
                raise
            
            # 3) Upload image to FAL storage and get URL (run in thread pool)
            try:
                print(f"[Seedance Video] Uploading image to FAL storage...")
                image_url = await asyncio.to_thread(fal_client.upload_file, temp_file_path)
                print(f"[Seedance Video] Image uploaded successfully: {image_url}")
                
                # Upload end_image if provided
                end_image_url = None
                if end_image_temp_path:
                    print(f"[Seedance Video] Uploading end_image to FAL storage...")
                    end_image_url = await asyncio.to_thread(fal_client.upload_file, end_image_temp_path)
                    print(f"[Seedance Video] End image uploaded successfully: {end_image_url}")
                    
            except Exception as e:
                print(f"[Seedance Video] Error uploading file: {str(e)}")
                # Clean up file before re-raising
                try:
                    os.remove(temp_file_path)
                    if end_image_temp_path:
                        os.remove(end_image_temp_path)
                except:
                    pass
                raise

            # 4) Build the input arguments
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "duration": duration,
                "camera_fixed": camera_fixed,
                "generate_audio": generate_audio,
                "enable_safety_checker": enable_safety_checker
            }
            
            if end_image_url:
                arguments["end_image_url"] = end_image_url
            
            # Add seed if it's not -1 (random)
            if seed != -1:
                arguments["seed"] = seed

            print(f"[Seedance Video] Starting video generation...")

            # 5) Call fal_client.subscribe() (run in thread pool for async execution)
            try:
                result = await asyncio.to_thread(
                    fal_client.subscribe,
                    "fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
                    arguments=arguments,
                    with_logs=True,
                    on_queue_update=self.on_queue_update,
                )
                print(f"[Seedance Video] Video generation completed successfully")
            except Exception as e:
                print(f"[Seedance Video] Error during video generation: {str(e)}")
                # Clean up file before re-raising
                try:
                    os.remove(temp_file_path)
                    if end_image_temp_path:
                        os.remove(end_image_temp_path)
                except:
                    pass
                raise

            # 6) Clean up temporary file
            os.remove(temp_file_path)
            if end_image_temp_path:
                os.remove(end_image_temp_path)
            print(f"[Seedance Video] Cleaned up temp files")

            if not result or not result.get("video"):
                raise ValueError("No valid video result from fal_client.subscribe().")

            # 7) Get the generated video URL
            video_data = result["video"]
            video_url = video_data["url"]
            seed_used = result.get("seed", "unknown")

            # 8) Download the generated video (run in thread pool)
            print(f"[Seedance Video] Downloading video from: {video_url}")
            resp = await asyncio.to_thread(requests.get, video_url)
            if resp.status_code != 200:
                raise ConnectionError(f"Failed to download video. HTTP {resp.status_code}")

            # 9) Save the video to temp directory & metadata (run in thread pool)
            temp_filename = self.get_temp_filename()
            
            generation_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
                "parameters": {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "duration": duration,
                    "camera_fixed": camera_fixed,
                    "generate_audio": generate_audio,
                    "enable_safety_checker": enable_safety_checker,
                    "seed": seed if seed != -1 else seed_used
                },
                "input_image_url": image_url,
                "output_video_info": video_data,
                "seed_used": seed_used,
                "fal_result": result
            }

            video_path, metadata_path = await asyncio.to_thread(
                self.save_video_and_metadata, resp.content, generation_info, temp_filename
            )
            print(f"[Seedance Video] Saved temp video -> {video_path}")
            print(f"[Seedance Video] Saved metadata -> {metadata_path}")

            # 10) Create VideoFromFile object for ComfyUI's VIDEO type
            video_output = VideoFromFile(video_path)
            
            # 11) Return the video object, video path string & metadata JSON string
            return (video_output, video_path, json.dumps(generation_info, indent=2))

        except Exception as e:
            # Return None for video, empty path and an error message in JSON
            error_info = {
                "error": f"Seedance video generation failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "fal-ai/bytedance/seedance/v1.5/pro/image-to-video"
            }
            print(f"[Seedance Video] Error: {str(e)}")
            return (None, empty_path, json.dumps(error_info, indent=2))

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
        print(f"[Seedance Video] Converting tensor to PIL - input shape: {tensor.shape}")
        
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # remove batch dimension
            print(f"[Seedance Video] Removed batch dimension, new shape: {tensor.shape}")

        arr = tensor.cpu().numpy()
        print(f"[Seedance Video] Converted to numpy array, shape: {arr.shape}")
        
        # If shape is (C, H, W), transpose it
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1, 2, 0))
            print(f"[Seedance Video] Transposed array, new shape: {arr.shape}")

        arr = (arr * 255).clip(0, 255).astype("uint8")
        print(f"[Seedance Video] Converted to uint8, shape: {arr.shape}")
        
        pil_img = Image.fromarray(arr)
        print(f"[Seedance Video] Created PIL image: {pil_img.size} mode: {pil_img.mode}")
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
        print("[Seedance Video] Interrupt called.")


# Register with ComfyUI
NODE_CLASS_MAPPINGS = {
    "FalAPI_seedance_video": FalAPI_seedance_video
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPI_seedance_video": "FAL Seedance Video Generation"
}

__all__ = ["FalAPI_seedance_video"]


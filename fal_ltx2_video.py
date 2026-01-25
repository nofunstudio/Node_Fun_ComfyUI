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

class FalAPI_ltx2_video:
    @classmethod
    def INPUT_TYPES(cls):
        """
        A video generation node for `fal-ai/ltx-2-19b/image-to-video/lora` and distilled variant.
        Takes an image input, prompt, and generates a video using LTX-2 Video API.
        """
        return {
            "required": {
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "display": "FAL API Token"
                }),
                "model_variant": (["regular", "distilled"], {
                    "default": "regular",
                    "display": "Model Variant"
                }),
                "image": ("IMAGE", {
                    "display": "Source Image"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic video transformation with smooth motion",
                    "display": "Video Prompt"
                }),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 1,
                    "max": 257,
                    "step": 1,
                    "display": "Number of Frames"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 8,
                    "display": "Width"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 8,
                    "display": "Height"
                }),
                "fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "FPS"
                }),
                "image_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "Image Strength"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake",
                    "display": "Negative Prompt"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "Guidance Scale (regular only)"
                }),
                "num_inference_steps": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "Inference Steps (regular only)"
                }),
                "use_multiscale": ("BOOLEAN", {
                    "default": True,
                    "display": "Use Multiscale"
                }),
                "generate_audio": ("BOOLEAN", {
                    "default": False,
                    "display": "Generate Audio"
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                    "display": "Enable Safety Checker"
                }),
                "acceleration": (["none", "tensorrt"], {
                    "default": "none",
                    "display": "Acceleration"
                }),
                "camera_lora": (["none", "dolly_in", "dolly_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "zoom_in", "zoom_out"], {
                    "default": "none",
                    "display": "Camera LoRA"
                }),
                "camera_lora_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "Camera LoRA Scale"
                }),
                "video_output_type": (["X264 (.mp4)", "H265 (.mp4)", "ProRes (.mov)", "GIF (.gif)"], {
                    "default": "X264 (.mp4)",
                    "display": "Video Output Type"
                }),
                "video_quality": (["low", "medium", "high", "ultra"], {
                    "default": "high",
                    "display": "Video Quality"
                }),
                "video_write_mode": (["fast", "balanced", "quality"], {
                    "default": "balanced",
                    "display": "Video Write Mode"
                }),
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
                "lora_name": ("STRING", {
                    "multiline": False,
                    "default": "LTX-2",
                    "display": "LoRA Weight Name"
                }),
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
        self.temp_dir = folder_paths.get_temp_directory()
        self.metadata_dir = os.path.join(self.temp_dir, "ltx2_metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

    def get_temp_filename(self, output_type):
        """Generate a unique temporary filename using timestamp and random component."""
        timestamp = int(time.time() * 1000)
        random_id = random.randint(1000, 9999)
        
        # Determine extension based on output type
        if "ProRes" in output_type:
            ext = "mov"
        elif "GIF" in output_type:
            ext = "gif"
        else:
            ext = "mp4"
        
        return f"ltx2_{timestamp}_{random_id}.{ext}"

    def save_video_and_metadata(self, video_content, generation_info, filename):
        """Saves the generated video to temp directory, plus metadata as a .json."""
        filepath = os.path.join(self.temp_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(video_content)

        metadata_filename = f"{os.path.splitext(filename)[0]}_metadata.json"
        metadata_filepath = os.path.join(self.metadata_dir, metadata_filename)

        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(generation_info, f, indent=4, ensure_ascii=False)

        return filepath, metadata_filepath

    def on_queue_update(self, update):
        """Handle queue updates and log messages"""
        try:
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(f"[LTX-2 Video] {log['message']}")
        except Exception as e:
            print(f"[LTX-2 Video] Error in queue update: {e}")

    def tensor_to_tempfile(self, tensor):
        """Convert a ComfyUI IMAGE tensor to a PNG file. Return the file path."""
        pil_img = self.tensor_to_pil(tensor)
        fd, filename = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil_img.save(filename, format="PNG")
        return filename

    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image: handle (B, H, W, C) or (C, H, W)."""
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        arr = tensor.cpu().numpy()
        
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1, 2, 0))

        arr = (arr * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)

    async def generate_video(self, api_token, model_variant, image, prompt, num_frames, width, height, fps, image_strength,
                             negative_prompt="", guidance_scale=3.0, num_inference_steps=40, use_multiscale=True,
                             generate_audio=False, enable_safety_checker=True, acceleration="none",
                             camera_lora="none", camera_lora_scale=1.0, video_output_type="X264 (.mp4)",
                             video_quality="high", video_write_mode="balanced", lora_url="", lora_scale=1.0,
                             lora_name="LTX-2", seed=-1):
        """Generate video using fal-ai/ltx-2-19b/image-to-video/lora (Async execution)"""
        
        print(f"[LTX-2 Video] Starting video generation with {model_variant} model...")
        
        empty_path = ""
        
        if not api_token:
            error_msg = "A FAL API token is required."
            print(f"[LTX-2 Video] Error: {error_msg}")
            raise ValueError(error_msg)
        
        if image is None:
            error_msg = "Image input is required."
            print(f"[LTX-2 Video] Error: {error_msg}")
            raise ValueError(error_msg)
            
        if not prompt or prompt.strip() == "":
            error_msg = "Prompt input is required."
            print(f"[LTX-2 Video] Error: {error_msg}")
            raise ValueError(error_msg)
            
        print(f"[LTX-2 Video] Input image shape: {image.shape}")
        print(f"[LTX-2 Video] Prompt: {prompt[:100]}...")
        print(f"[LTX-2 Video] Settings: {num_frames} frames, {width}x{height}, {fps} fps")

        temp_file_path = None
        
        try:
            os.environ["FAL_KEY"] = api_token
            print(f"[LTX-2 Video] API token set, processing image...")

            # Convert image to temp file and upload
            temp_file_path = await asyncio.to_thread(self.tensor_to_tempfile, image)
            print(f"[LTX-2 Video] Created temp file: {temp_file_path}")
            
            print(f"[LTX-2 Video] Uploading image to FAL storage...")
            image_url = await asyncio.to_thread(fal_client.upload_file, temp_file_path)
            print(f"[LTX-2 Video] Image uploaded successfully: {image_url}")

            # Build arguments
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "num_frames": num_frames,
                "video_size": {"width": width, "height": height},
                "fps": fps,
                "image_strength": image_strength,
                "use_multiscale": use_multiscale,
                "generate_audio": generate_audio,
                "enable_safety_checker": enable_safety_checker,
                "acceleration": acceleration,
                "camera_lora": camera_lora,
                "camera_lora_scale": camera_lora_scale,
                "video_output_type": video_output_type,
                "video_quality": video_quality,
                "video_write_mode": video_write_mode,
            }

            # Add negative prompt if provided
            if negative_prompt.strip():
                arguments["negative_prompt"] = negative_prompt

            # Add guidance_scale and num_inference_steps only for regular model
            if model_variant == "regular":
                arguments["guidance_scale"] = guidance_scale
                arguments["num_inference_steps"] = num_inference_steps

            # Add LoRA if provided
            if lora_url.strip():
                arguments["loras"] = [{
                    "path": lora_url,
                    "scale": lora_scale,
                    "weight_name": lora_name
                }]

            # Add seed if not random
            if seed != -1:
                arguments["seed"] = seed

            # Select endpoint based on model variant
            if model_variant == "distilled":
                endpoint = "fal-ai/ltx-2-19b/distilled/image-to-video/lora"
            else:
                endpoint = "fal-ai/ltx-2-19b/image-to-video/lora"

            print(f"[LTX-2 Video] Submitting to {endpoint}...")
            print(f"[LTX-2 Video] Arguments: {json.dumps({k: v for k, v in arguments.items() if k != 'negative_prompt'}, indent=2)}")

            # Call API
            result = await asyncio.to_thread(
                fal_client.subscribe,
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=self.on_queue_update,
            )
            print(f"[LTX-2 Video] Video generation completed successfully")

            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"[LTX-2 Video] Cleaned up temp file")

            if not result or not result.get("video"):
                raise ValueError("No valid video result from fal_client.subscribe().")

            # Get the generated video URL
            video_data = result["video"]
            video_url = video_data["url"]
            seed_used = result.get("seed", "unknown")

            # Download the generated video
            print(f"[LTX-2 Video] Downloading video from: {video_url}")
            resp = await asyncio.to_thread(requests.get, video_url)
            if resp.status_code != 200:
                raise ConnectionError(f"Failed to download video. HTTP {resp.status_code}")

            # Save the video to temp directory & metadata
            temp_filename = self.get_temp_filename(video_output_type)
            
            generation_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": endpoint,
                "model_variant": model_variant,
                "parameters": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_frames": num_frames,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "image_strength": image_strength,
                    "guidance_scale": guidance_scale if model_variant == "regular" else None,
                    "num_inference_steps": num_inference_steps if model_variant == "regular" else None,
                    "use_multiscale": use_multiscale,
                    "generate_audio": generate_audio,
                    "acceleration": acceleration,
                    "camera_lora": camera_lora,
                    "camera_lora_scale": camera_lora_scale,
                    "video_output_type": video_output_type,
                    "video_quality": video_quality,
                    "video_write_mode": video_write_mode,
                    "lora_url": lora_url if lora_url.strip() else None,
                    "lora_scale": lora_scale if lora_url.strip() else None,
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
            print(f"[LTX-2 Video] Saved temp video -> {video_path}")
            print(f"[LTX-2 Video] Saved metadata -> {metadata_path}")

            # Create VideoFromFile object for ComfyUI's VIDEO type
            video_output = VideoFromFile(video_path)
            
            return (video_output, video_path, json.dumps(generation_info, indent=2))

        except Exception as e:
            # Clean up temp file on error
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            
            error_info = {
                "error": f"LTX-2 video generation failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": f"fal-ai/ltx-2-19b/{model_variant}/image-to-video/lora"
            }
            print(f"[LTX-2 Video] Error: {str(e)}")
            return (None, empty_path, json.dumps(error_info, indent=2))

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def interrupt(self):
        print("[LTX-2 Video] Interrupt called.")


NODE_CLASS_MAPPINGS = {
    "FalAPI_ltx2_video": FalAPI_ltx2_video
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPI_ltx2_video": "FAL LTX-2 Video Generation"
}

__all__ = ["FalAPI_ltx2_video"]

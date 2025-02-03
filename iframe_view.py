import os
import PIL
from PIL import Image, ImageOps
import hashlib
import base64
from io import BytesIO
import torch
import numpy as np
import time
import json
import traceback
from aiohttp import web
import folder_paths
import asyncio


DEBUG = True
THREEVIEW_DICT = {}  # Global dict (can be retained for compatibility if needed)


# Global dictionary to store texture data (populated from the JS side via postMessage)
TEXTURE_STORE = {}

def get_global_texture_value(unique_id, key):
    if unique_id in TEXTURE_STORE:
        return TEXTURE_STORE[unique_id].get(key, None)
    return None


def process_texture_data(data):
    """
    Process incoming texture data from the iframe.
    We expect data to be a dict containing a base64-encoded image string (under the "base64" key)
    along with width, height, and other metadata.
    """
    start_time = time.time()
    print("[process_texture_data] Started processing texture data. Data type:", type(data))
    
    if data is None:
        print("[process_texture_data] No texture data received (data is None).")
        result = torch.zeros((64, 64, 3), dtype=torch.float32)
        print(f"[process_texture_data] Returning default tensor with shape: {result.shape}")
        return result

    # Log the structure of the data if it's a dict.
    if isinstance(data, dict):
        print("[process_texture_data] Texture data keys:", list(data.keys()))
        if "base64" in data:
            base64_string = data.get("base64")
            print("[process_texture_data] Found 'base64' key with string length:", len(base64_string))
        elif "data" in data:
            base64_string = data.get("data")
            print("[process_texture_data] Found 'data' key with string length:", len(base64_string) if isinstance(base64_string, str) else "non-string")
    else:
        print("[process_texture_data] Unexpected data format. Data:", data)
        
    try:
        # Process if the data dict has either "base64" or "data"
        if isinstance(data, dict) and ("data" in data or "base64" in data):
            b64data = data.get("base64", data.get("data"))
            if isinstance(b64data, str):
                print("[process_texture_data] b64data is string. Checking for header...")
                if b64data.startswith("data:"):
                    print("[process_texture_data] Stripping data URL header from base64 string.")
                    parts = b64data.split(",")
                    print("[process_texture_data] Header part:", parts[0])
                    b64data = parts[-1]
                print("[process_texture_data] Final base64 string length:", len(b64data))
                image_bytes = base64.b64decode(b64data)
                print("[process_texture_data] Decoded base64, received", len(image_bytes), "bytes.")
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                print("[process_texture_data] PIL image mode:", pil_image.mode)
                image = np.array(pil_image)
                print(f"[process_texture_data] PIL image shape: {image.shape}")
            else:
                print("[process_texture_data] b64data is not a string. Type:", type(b64data))
                width = data.get("width", 64)
                height = data.get("height", 64)
                pixel_array = np.array(data.get("data"), dtype=np.uint8)
                print("[process_texture_data] Raw pixel_array shape from 'data':", pixel_array.shape)
                try:
                    image = pixel_array.reshape(height, width, 3)
                    print(f"[process_texture_data] Reshaped image to: {image.shape}")
                except Exception as reshape_err:
                    print(f"[process_texture_data] Error reshaping array: {reshape_err}")
                    raise
        else:
            print("[process_texture_data] Data format not recognized, returning default tensor.")
            result = torch.zeros((64, 64, 3), dtype=torch.float32)
            print(f"[process_texture_data] Returning default tensor with shape: {result.shape}")
            return result
            
        print(f"[process_texture_data] Post-conversion image stats: shape={image.shape}, min={image.min()}, max={image.max()}")
        image = image.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(image)
        print(f"[process_texture_data] Final tensor shape: {tensor.shape}, range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
        elapsed = time.time() - start_time
        print(f"[process_texture_data] Completed processing in {elapsed:.4f} seconds.")
        return tensor
    except Exception as e:
        print(f"[process_texture_data] Error processing texture data: {e}")
        traceback.print_exc()
        result = torch.zeros((64, 64, 3), dtype=torch.float32)
        return result


async def wait_for_capture_completion(unique_id, target_frame_count, timeout=1.0, interval=0.1):
    waited = 0.0
    while waited < timeout:
        frames = get_global_texture_value(unique_id, "animationFrames")
        frame_count = len(frames) if frames is not None else 0
        print(f"[wait_for_capture_completion] {waited:.2f}s elapsed: Found {frame_count} frames for unique_id '{unique_id}'")
        if frames and frame_count >= target_frame_count:
            return frames
        await asyncio.sleep(interval)
        waited += interval
    print("[wait_for_capture_completion] Timeout reached, returning empty list.")
    return []

class IframeView:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "scene_state": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "{\"camera\": {\"position\": [0,0,5]}, \"animation\": false}",
                    },
                ),
                "frame_count": ("INT", {"default": 4, "min": 1, "max": 60}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "animationFrames": "TEXTURE_SEQUENCE",
            },
        }

    # Only return animation textures now.
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("animation_color", "animation_canny", "animation_depth", "animation_normal")
    FUNCTION = "process_iframe"
    CATEGORY = "lth"

    async def _process_iframe_async(
        self,
        url,
        width,
        height,
        scene_state,
        frame_count,
        unique_id,
        animationFrames=None,
    ):
        start_time = time.time()
        print("[process_iframe] Starting asynchronous processing at", start_time)
        unique_id = str(unique_id)

        if animationFrames is not None and len(animationFrames) >= frame_count:
            print("[process_iframe] Using provided animationFrames from hidden data.")
        else:
            print("[process_iframe] Animation frames not provided or incomplete; using fallback.")
            animationFrames = await wait_for_capture_completion(unique_id, frame_count)

        # Process each frame's texture into a tensor for animation outputs.
        color_frames = [
            process_texture_data(frame["textures"]["color"])
            for frame in animationFrames
            if "textures" in frame and "color" in frame["textures"]
        ]
        canny_frames = [
            process_texture_data(frame["textures"]["canny"])
            for frame in animationFrames
            if "textures" in frame and "canny" in frame["textures"]
        ]
        depth_frames = [
            process_texture_data(frame["textures"]["depth"])
            for frame in animationFrames
            if "textures" in frame and "depth" in frame["textures"]
        ]
        normal_frames = [
            process_texture_data(frame["textures"]["normal"])
            for frame in animationFrames
            if "textures" in frame and "normal" in frame["textures"]
        ]

        animation_color   = torch.stack(color_frames, dim=0)  if color_frames else None
        animation_canny   = torch.stack(canny_frames, dim=0)  if canny_frames else None
        animation_depth   = torch.stack(depth_frames, dim=0)  if depth_frames else None
        animation_normal  = torch.stack(normal_frames, dim=0) if normal_frames else None

        # Clear the heavy animationFrames data.
        animationFrames = None

        elapsed = time.time() - start_time
        print(f"[process_iframe] Asynchronous processing completed in {elapsed:.4f} seconds.")
        return (animation_color, animation_canny, animation_depth, animation_normal)

    def process_iframe(
        self,
        url,
        width,
        height,
        scene_state,
        frame_count,
        unique_id,
        animationFrames=None,
    ):
        return asyncio.run(
            self._process_iframe_async(
                url,
                width,
                height,
                scene_state,
                frame_count,
                unique_id,
                animationFrames,
            )
        )

NODE_CLASS_MAPPINGS = {
    "IframeView": IframeView
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IframeView": "Iframe View"
}

WEB_DIRECTORY = "./js"

GREEN = "\033[92m"
LIGHT_YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
CLEAR = "\033[0m"

nodesNames = ", ".join(NODE_DISPLAY_NAME_MAPPINGS.values())
print(f"\n{MAGENTA}* {GREEN}lo-th -> {LIGHT_YELLOW}{nodesNames} {BLUE}<Loaded>{CLEAR}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

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
    start_time = time.time()
    print("[process_texture_data] Started processing texture data.")
    
    if not data:
        print("[process_texture_data] No texture data received.")
        result = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        print(f"[process_texture_data] Returning default tensor with shape: {result.shape}")
        return result

    try:
        print("Raw data received:")
        if isinstance(data, dict) and "data" in data:
            if isinstance(data["data"], str):
                print("[process_texture_data] Received base64 encoded texture data.")
                b64data = data["data"]
                if b64data.startswith("data:"):
                    print("[process_texture_data] Stripping data URL header from base64 string.")
                    b64data = b64data.split(",")[-1]
                image_bytes = base64.b64decode(b64data)
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image = np.array(pil_image)
                print(f"[process_texture_data] PIL image shape: {image.shape}")
            else:
                print("[process_texture_data] Received raw numeric texture data.")
                width = data.get("width", 64)
                height = data.get("height", 64)
                pixel_array = np.array(data["data"], dtype=np.uint8)
                try:
                    image = pixel_array.reshape(height, width, 3)
                    print(f"[process_texture_data] Reshaped image to: {image.shape}")
                except Exception as reshape_err:
                    print(f"[process_texture_data] Error reshaping array: {reshape_err}")
                    raise
        else:
            print("[process_texture_data] Data format not recognized, returning default.")
            result = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            print(f"[process_texture_data] Returning default tensor with shape: {result.shape}")
            return result
            
        print(f"[process_texture_data] Post reshape image stats: shape={image.shape}, min={image.min()}, max={image.max()}")
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image)[None,]
        print(f"[process_texture_data] Tensor shape: {tensor.shape}, range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
        elapsed = time.time() - start_time
        print(f"[process_texture_data] Completed processing in {elapsed:.4f} seconds.")
        return tensor
    except Exception as e:
        print(f"[process_texture_data] Error processing texture data: {e}")
        traceback.print_exc()
        result = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return result

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
                    {"multiline": True, "default": "{\"camera\": {\"position\": [0,0,5]}, \"animation\": false}"}
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "color": "TEXTURE",
                "canny": "TEXTURE",
                "depth": "TEXTURE",
                "normal": "TEXTURE"
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("color", "canny", "depth", "normal")
    FUNCTION = "process_iframe"
    CATEGORY = "lth"

    async def _process_iframe_async(self, url, width, height, scene_state, unique_id, color=None, canny=None, depth=None, normal=None):
        start_time = time.time()
        print("[process_iframe] Starting asynchronous processing at", start_time)
        try:
            print("\n[process_iframe] Called with inputs:")
            print("  url:", url)
            print("  width:", width, "height:", height)
            print("  scene_state:", scene_state)
            print("  unique_id:", unique_id)

            # Since the iframe is always sending fully processed texture data,
            # there is no need to trigger an intermediate save check.
            # Directly retrieve the texture data.
            if color is None:
                color = get_global_texture_value(unique_id, "color")
            if canny is None:
                canny = get_global_texture_value(unique_id, "canny")
            if depth is None:
                depth = get_global_texture_value(unique_id, "depth")
            if normal is None:
                normal = get_global_texture_value(unique_id, "normal")

            try:
                state_data = json.loads(scene_state)
            except json.JSONDecodeError as e:
                print(f"[process_iframe] Invalid scene state JSON: {e}")
                state_data = {}

            color_tensor = process_texture_data(color)
            canny_tensor = process_texture_data(canny)
            depth_tensor = process_texture_data(depth)
            normal_tensor = process_texture_data(normal)
            
            elapsed = time.time() - start_time
            print(f"[process_iframe] Asynchronous processing completed in {elapsed:.4f} seconds.")
            return (color_tensor, canny_tensor, depth_tensor, normal_tensor)
        except Exception as e:
            print(f"[process_iframe] Error: {e}")
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, empty, empty, empty)

    def process_iframe(self, url, width, height, scene_state, unique_id, color=None, canny=None, depth=None, normal=None):
        """
        Synchronous wrapper for _process_iframe_async.
        """
        return asyncio.run(
            self._process_iframe_async(
                url, width, height, scene_state, unique_id, color, canny, depth, normal
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

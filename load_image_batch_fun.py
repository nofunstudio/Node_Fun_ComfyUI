import os
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
import hashlib
import folder_paths

class LoadImageBatchFun:
    """
    A fixed version of Load Image Batch that supports multiple instances in the same workflow.
    Each instance maintains its own state to prevent conflicts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "random"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": "Batch"}),
                "path": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": "*", "multiline": False}),
                "allow_RGBA_output": (["false", "true"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "filename", "count")
    FUNCTION = "load_batch_images"
    CATEGORY = "image"

    def __init__(self):
        """
        Initialize instance-specific variables to prevent shared state issues.
        This is the key fix for supporting multiple instances.
        """
        self.current_index = 0
        self.last_hash = None
        self.cached_filenames = []

    def load_batch_images(self, mode, index, label, path, pattern, allow_RGBA_output):
        """
        Load images from a directory path based on the specified mode.
        """
        # Validate path
        if not path or path.strip() == "":
            raise ValueError("Path cannot be empty")
        
        # Expand path and validate
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")
        
        if not os.path.isdir(path):
            raise ValueError(f"Path is not a directory: {path}")
        
        # Generate a hash for path/pattern to determine if we need to reload images
        # Don't include index/mode in this hash so we can reuse cached image paths
        cache_hash = hashlib.md5(
            f"{path}_{pattern}".encode()
        ).hexdigest()
        
        # Load image paths if cache is invalid or parameters changed
        if self.last_hash != cache_hash or not self.cached_filenames:
            print(f"[Load Image Batch Fun] Loading image paths from: {path}")
            self.cached_filenames = self._load_images_from_path(
                path, pattern
            )
            self.last_hash = cache_hash
            self.current_index = 0
        
        if not self.cached_filenames:
            raise ValueError(f"No images found in path: {path} with pattern: {pattern}")
        
        total_images = len(self.cached_filenames)
        print(f"[Load Image Batch Fun] Total images found: {total_images}")
        
        # Select image path based on mode
        selected_path = None
        if mode == "single_image":
            # Use the specified index, wrap around if needed
            selected_index = index % total_images
            selected_path = self.cached_filenames[selected_index]
            print(f"[Load Image Batch Fun] Single image mode - selected index: {selected_index}")
            
        elif mode == "incremental_image":
            # Use current index and increment
            selected_index = self.current_index % total_images
            selected_path = self.cached_filenames[selected_index]
            self.current_index += 1
            print(f"[Load Image Batch Fun] Incremental mode - selected index: {selected_index}, next: {self.current_index}")
            
        elif mode == "random":
            # Random selection
            import random
            selected_index = random.randint(0, total_images - 1)
            selected_path = self.cached_filenames[selected_index]
            print(f"[Load Image Batch Fun] Random mode - selected index: {selected_index}")
        
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Load the selected image
        selected_image = self._load_image(selected_path, allow_RGBA_output)
        
        # Get filename without extension
        selected_filename = os.path.splitext(os.path.basename(selected_path))[0]
        
        return (selected_image, selected_filename, total_images)

    def _load_images_from_path(self, path, pattern):
        """
        Get all image file paths from the specified path matching the pattern.
        Returns a sorted list of full file paths.
        """
        import glob
        
        # Get all files matching pattern
        if pattern == "*":
            search_pattern = os.path.join(path, "*")
        else:
            search_pattern = os.path.join(path, pattern)
        
        files = glob.glob(search_pattern)
        
        # Filter for image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        image_files = [
            f for f in files 
            if os.path.isfile(f) and os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        # Sort files for consistent ordering
        image_files.sort()
        
        if not image_files:
            return []
        
        print(f"[Load Image Batch Fun] Found {len(image_files)} image files")
        
        return image_files

    def _load_image(self, image_path, allow_RGBA_output):
        """
        Load a single image and convert it to the ComfyUI tensor format.
        """
        img = Image.open(image_path)
        
        # Handle animated images (take first frame)
        if hasattr(img, 'is_animated') and img.is_animated:
            img = ImageSequence.Iterator(img)[0]
        
        # Handle EXIF orientation
        img = ImageOps.exif_transpose(img)
        
        # Convert to RGB or RGBA based on settings
        if allow_RGBA_output == "true" and img.mode == "RGBA":
            # Keep RGBA
            img_array = np.array(img).astype(np.float32) / 255.0
        else:
            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor with shape (1, H, W, C)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        return img_tensor

    @classmethod
    def IS_CHANGED(cls, mode, index, label, path, pattern, allow_RGBA_output):
        """
        This tells ComfyUI when to re-execute the node.
        We return a hash of the parameters to detect changes.
        """
        # For single_image mode, include index in the hash so changing index triggers re-execution
        # For incremental_image mode, always return a new value so it increments each time
        # For random mode, always return a new value so it randomizes each time
        
        if mode == "incremental_image" or mode == "random":
            # Always execute for these modes
            import time
            return str(time.time())
        else:
            # For single_image mode, return hash of all parameters including index
            param_string = f"{mode}_{index}_{label}_{path}_{pattern}_{allow_RGBA_output}"
            return hashlib.md5(param_string.encode()).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, mode, index, label, path, pattern, allow_RGBA_output):
        """
        Validate inputs before execution.
        """
        if not path or path.strip() == "":
            return True
        
        expanded_path = os.path.expanduser(path)
        if not os.path.exists(expanded_path):
            return f"Path does not exist: {path}"
        
        if not os.path.isdir(expanded_path):
            return f"Path is not a directory: {path}"
        
        return True


# Register with ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadImageBatchFun": LoadImageBatchFun
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageBatchFun": "Load Image Batch (Fun)"
}

__all__ = ["LoadImageBatchFun"]


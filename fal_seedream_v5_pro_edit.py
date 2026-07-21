import os
import re
import time
import requests
import tempfile
import json
import numpy as np
import torch
import asyncio
import folder_paths
import fal_client
from io import BytesIO
from PIL import Image


# Match ByteDance Seedream 5.0 Pro partner node size presets
SIZE_PRESETS = [
    "Custom",
    "(1K) 1024x1024 (1:1)",
    "(1K) 864x1152 (3:4)",
    "(1K) 1152x864 (4:3)",
    "(1K) 1312x736 (16:9)",
    "(1K) 736x1312 (9:16)",
    "(1K) 832x1248 (2:3)",
    "(1K) 1248x832 (3:2)",
    "(1K) 1568x672 (21:9)",
    "(2K) 2048x2048 (1:1)",
    "(2K) 1728x2304 (3:4)",
    "(2K) 2304x1728 (4:3)",
    "(2K) 1664x2496 (2:3)",
    "(2K) 2496x1664 (3:2)",
]

ENDPOINT = "bytedance/seedream/v5/pro/edit"


def parse_size_preset(preset):
    """Extract width/height from a preset label like '(2K) 2048x2048 (1:1)'."""
    match = re.search(r"(\d+)x(\d+)", preset)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


class FalAPI_SeedreamV5ProEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("FAL_KEY", ""),
                    "display": "FAL API Key"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Edit the image according to the prompt.",
                    "display": "Prompt"
                }),
                "image1": ("IMAGE", {"display": "Image 1"}),
                "size_preset": (SIZE_PRESETS, {
                    "default": "Custom",
                    "display": "Size Preset"
                }),
                "width": ("INT", {
                    "default": 2048,
                    "min": 1024,
                    "max": 3136,
                    "step": 1,
                    "display": "Width (Custom)"
                }),
                "height": ("INT", {
                    "default": 2048,
                    "min": 1024,
                    "max": 2496,
                    "step": 1,
                    "display": "Height (Custom)"
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "display": "Num Images"
                }),
            },
            "optional": {
                "image2": ("IMAGE", {"display": "Image 2 (optional)"}),
                "image3": ("IMAGE", {"display": "Image 3 (optional)"}),
                "image4": ("IMAGE", {"display": "Image 4 (optional)"}),
                "image5": ("IMAGE", {"display": "Image 5 (optional)"}),
                "image6": ("IMAGE", {"display": "Image 6 (optional)"}),
                "image7": ("IMAGE", {"display": "Image 7 (optional)"}),
                "image8": ("IMAGE", {"display": "Image 8 (optional)"}),
                "image9": ("IMAGE", {"display": "Image 9 (optional)"}),
                "image10": ("IMAGE", {"display": "Image 10 (optional)"}),
                "output_format": (["jpeg", "png"], {
                    "default": "png",
                    "display": "Output Format"
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                    "display": "Enable Safety Checker"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("images", "image_paths", "generation_info", "generation_time",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL"

    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
        self.output_dir = os.path.join("output", "API", "FAL", "seedream-v5-pro-edit")
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def get_next_number(self):
        valid_exts = {".png", ".jpg", ".jpeg"}
        files = [
            f for f in os.listdir(self.output_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        numbers = []
        for file_name in files:
            base, _ = os.path.splitext(file_name)
            try:
                numbers.append(int(base))
            except ValueError:
                pass
        return max(numbers) + 1 if numbers else 1

    def save_image_and_metadata(self, pil_img, generation_info, number, ext="png"):
        filename = f"{number:03d}.{ext}"
        filepath = os.path.join(self.output_dir, filename)
        pil_img.save(filepath)

        metadata_filename = f"{number:03d}_metadata.json"
        metadata_filepath = os.path.join(self.metadata_dir, metadata_filename)
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(generation_info, f, indent=4, ensure_ascii=False)
        return filepath, metadata_filepath

    def on_queue_update(self, update):
        try:
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(f"[SeedreamV5ProEdit] {log['message']}")
        except Exception as e:
            print(f"[SeedreamV5ProEdit] Error in queue update: {e}")

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

    def pil_to_tensor(self, pil_img):
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]

    def is_empty_image(self, tensor):
        if tensor is None:
            return True
        return tensor.mean().item() < 0.01

    def resolve_image_size(self, size_preset, width, height):
        if size_preset != "Custom":
            parsed = parse_size_preset(size_preset)
            if parsed:
                width, height = parsed
        return {"width": int(width), "height": int(height)}

    def format_generation_time(self, elapsed_seconds):
        seconds = int(elapsed_seconds)
        centiseconds = int((elapsed_seconds - seconds) * 100)
        return f"{seconds}-{centiseconds:02d}"

    async def generate_image(
        self,
        api_key,
        prompt,
        image1,
        size_preset,
        width,
        height,
        num_images,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
        output_format="png",
        enable_safety_checker=True,
    ):
        print("[SeedreamV5ProEdit] Starting edit...")
        start_time = time.time()

        if not api_key:
            raise ValueError("A FAL API key is required.")

        os.environ["FAL_KEY"] = api_key
        temp_files = []

        try:
            image_urls = []
            all_images = [
                image1, image2, image3, image4, image5,
                image6, image7, image8, image9, image10,
            ]

            for i, img_tensor in enumerate(all_images):
                if img_tensor is not None and not self.is_empty_image(img_tensor):
                    print(f"[SeedreamV5ProEdit] Uploading image {i + 1}...")
                    temp_path = await asyncio.to_thread(self.tensor_to_tempfile, img_tensor)
                    temp_files.append(temp_path)
                    url = await asyncio.to_thread(fal_client.upload_file, temp_path)
                    image_urls.append(url)

            if not image_urls:
                raise ValueError("At least one image input is required.")

            # API keeps only the last 10 if more are sent
            image_urls = image_urls[-10:]
            image_size = self.resolve_image_size(size_preset, width, height)

            arguments = {
                "prompt": prompt,
                "image_urls": image_urls,
                "image_size": image_size,
                "num_images": num_images,
                "output_format": output_format,
                "enable_safety_checker": enable_safety_checker,
                "sync_mode": False,
            }

            print(f"[SeedreamV5ProEdit] Arguments: {json.dumps(arguments, indent=2)}")
            print(f"[SeedreamV5ProEdit] Submitting to {ENDPOINT}...")

            result = await asyncio.to_thread(
                fal_client.subscribe,
                ENDPOINT,
                arguments=arguments,
                with_logs=True,
                on_queue_update=self.on_queue_update,
            )

            if not result:
                raise RuntimeError("No result returned from API")

            output_images = result.get("images", [])
            if not output_images:
                raise RuntimeError(f"No images in result: {list(result.keys())}")

            print(f"[SeedreamV5ProEdit] Got {len(output_images)} image(s).")

            final_tensors = []
            saved_paths = []
            ext = "jpg" if output_format == "jpeg" else "png"

            for img_info in output_images:
                img_url = img_info.get("url")
                if not img_url:
                    continue

                resp = await asyncio.to_thread(requests.get, img_url)
                if resp.status_code != 200:
                    continue

                pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
                final_tensors.append(self.pil_to_tensor(pil_img))

                number = self.get_next_number()
                gen_info = {
                    "prompt": prompt,
                    "size_preset": size_preset,
                    "parameters": arguments,
                    "result": result,
                }
                path, _ = await asyncio.to_thread(
                    self.save_image_and_metadata, pil_img, gen_info, number, ext
                )
                saved_paths.append(path)

            if not final_tensors:
                raise RuntimeError("Failed to process output images")

            batch_tensor = (
                torch.cat(final_tensors, dim=0)
                if len(final_tensors) > 1
                else final_tensors[0]
            )
            generation_time = self.format_generation_time(time.time() - start_time)
            print(f"[SeedreamV5ProEdit] Done in {generation_time}s")
            return (
                batch_tensor,
                ";".join(saved_paths),
                json.dumps(result, indent=2),
                generation_time,
            )

        except Exception as e:
            print(f"[SeedreamV5ProEdit] Error: {e}")
            error_info = {
                "error": f"SeedreamV5ProEdit failed: {e}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            generation_time = self.format_generation_time(time.time() - start_time)
            return (
                torch.zeros((1, 64, 64, 3)),
                "",
                json.dumps(error_info, indent=2),
                generation_time,
            )

        finally:
            for f in temp_files:
                try:
                    os.remove(f)
                except OSError:
                    pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "FalAPI_SeedreamV5ProEdit": FalAPI_SeedreamV5ProEdit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPI_SeedreamV5ProEdit": "FAL Seedream 5.0 Pro Edit"
}

__all__ = ["FalAPI_SeedreamV5ProEdit"]

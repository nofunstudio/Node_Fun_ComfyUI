from PIL import Image
import numpy as np
import torch

def tensor_to_pil(tensor):
    """Convert ComfyUI tensor to PIL Image"""
    # ComfyUI tensors are typically in format [batch, height, width, channels]
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Convert from tensor to numpy array and scale to 0-255
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image, 'RGB')

def pil_to_tensor(pil_image):
    """Convert PIL Image to ComfyUI tensor"""
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_image)
    return tensor.unsqueeze(0)  # Add batch dimension

class MultiAlphaComposite:
    CATEGORY = "Image/Composite"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_images": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
            "optional": {
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    def composite(self, num_images, image1, image2, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None):
        # Collect all provided images up to num_images
        images = [image1, image2, image3, image4, image5, image6, image7, image8]
        valid_images = [img for img in images[:num_images] if img is not None]
        
        if len(valid_images) < 2:
            raise ValueError("At least 2 images are required for compositing")
        
        # Convert ComfyUI tensors to PIL Images
        pil_images = [tensor_to_pil(img) for img in valid_images]
        
        # Convert to RGBA for alpha compositing
        layers = [img.convert("RGBA") for img in pil_images]
        
        # Composite layers
        base = layers[0]
        for layer in layers[1:]:
            # Resize layer to match base if needed
            if layer.size != base.size:
                layer = layer.resize(base.size, Image.LANCZOS)
            base = Image.alpha_composite(base, layer)
        
        # Convert back to ComfyUI tensor format
        result_tensor = pil_to_tensor(base.convert("RGB"))
        return (result_tensor,) 
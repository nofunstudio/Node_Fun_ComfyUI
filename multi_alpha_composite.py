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
    
    # Handle different channel counts
    if np_image.shape[2] == 4:  # RGBA
        return Image.fromarray(np_image, 'RGBA')
    elif np_image.shape[2] == 3:  # RGB
        return Image.fromarray(np_image, 'RGB')
    else:  # Grayscale or other
        return Image.fromarray(np_image[:,:,0], 'L').convert('RGBA')

def pil_to_tensor(pil_image):
    """Convert PIL Image to ComfyUI tensor"""
    # Convert to RGB for output (ComfyUI typically expects RGB)
    if pil_image.mode == 'RGBA':
        # Create a white background and composite the RGBA image onto it
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        pil_image = background
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
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
        
        print(f"Debug: Processing {len(valid_images)} images for layer compositing")
        
        # Convert ComfyUI tensors to PIL Images and ensure RGBA
        layers = []
        for i, img_tensor in enumerate(valid_images):
            pil_img = tensor_to_pil(img_tensor)
            # Convert to RGBA for alpha compositing (like Photoshop layers)
            rgba_layer = pil_img.convert('RGBA')
            layers.append(rgba_layer)
            print(f"Debug: Layer {i+1} - Size: {rgba_layer.size}, Mode: {rgba_layer.mode}")
        
        # Start with the bottom layer (image1)
        result = layers[0]
        print(f"Debug: Starting with base layer (image1)")
        
        # Stack each subsequent layer on top (like Photoshop)
        for i in range(1, len(layers)):
            print(f"Debug: Compositing layer {i+1} on top")
            result = Image.alpha_composite(result, layers[i])
        
        print(f"Debug: Final composite size: {result.size}, mode: {result.mode}")
        
        # Convert back to ComfyUI tensor format
        result_tensor = pil_to_tensor(result)
        return (result_tensor,) 
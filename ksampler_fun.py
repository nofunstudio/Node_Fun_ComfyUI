from __future__ import annotations
import torch
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import latent_preview
import node_helpers
import os
import folder_paths
import numpy as np
from PIL import Image

def fun_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, vae, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, preview_steps=5):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    # Calculate which steps to capture previews
    preview_step_interval = max(1, steps // preview_steps)
    preview_steps_list = list(range(0, steps, preview_step_interval))
    if steps - 1 not in preview_steps_list:
        preview_steps_list.append(steps - 1)

    # Create a custom callback to capture intermediate images
    class PreviewCallback:
        def __init__(self, latent, preview_steps_list, vae):
            self.latent = latent
            self.preview_steps_list = preview_steps_list
            self.vae = vae
            self.output_dir = os.path.join(folder_paths.get_output_directory(), "ksampler_previews")
            os.makedirs(self.output_dir, exist_ok=True)

        def __call__(self, step, denoised, x, total_steps):
            if step in self.preview_steps_list:
                # Create a copy of the current state
                current_latent = self.latent.copy()
                current_latent["samples"] = x
                
                # Decode the latent to image
                with torch.no_grad():
                    image = self.vae.decode(x)
                    image = image.cpu().numpy()
                    image = (image * 255).astype(np.uint8)
                    image = Image.fromarray(image[0])
                    
                    # Save the image
                    filename = f"preview_step_{step:03d}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    image.save(filepath)

    # Set up the callback with VAE
    callback = PreviewCallback(latent, preview_steps_list, vae)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    
    out = latent.copy()
    out["samples"] = samples
    return (out, None)  # Return None for preview_latents since we're saving directly

class KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "vae": ("VAE", {"tooltip": "The VAE used for decoding the latent images."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "preview_steps": ("INT", {"default": 5, "min": 1, "max": 20, "tooltip": "Number of intermediate previews to generate during sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "preview_latents")
    OUTPUT_TOOLTIPS = ("The final denoised latent.", "Preview images are saved to the outputs/ksampler_previews folder.")
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image. Saves intermediate previews to the outputs folder."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, vae, denoise=1.0, preview_steps=5):
        return fun_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, vae, denoise=denoise, preview_steps=preview_steps)

NODE_CLASS_MAPPINGS = {
    "Fun KSampler": KSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fun KSampler": "Fun KSampler"
}

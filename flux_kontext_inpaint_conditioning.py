# Copyright 2025 ZenAI, Inc.
# Author: @Trgtuan_10, @vuongminh1907 (modified with padding fix)
import torch
from comfy.comfy_types import IO
import node_helpers

class FluxKontextInpaintingConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning":      ("CONDITIONING", ),
                "vae":               ("VAE", ),
                "pixels":            ("IMAGE", ),
                "mask":              ("MASK", ),
                "reference_latents": ("LATENT", ),  # face-latent port
                "noise_mask":        ("BOOLEAN", {
                                         "default": True,
                                         "tooltip": "Restrict noise to the mask region."
                                     }),
            }
        }

    RETURN_TYPES = ("CONDITIONING","LATENT")
    RETURN_NAMES = ("conditioning", "latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/inpaint"

    def _append_reference_latent(self, conditioning, ref_latent):
        """Append the face-reference latent to conditioning."""
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": [ref_latent["samples"]]},
                append=True
            )
        return conditioning

    def encode(self, conditioning, pixels, vae, mask, reference_latents, noise_mask=True):
        # 1) Upsample mask to image size with nearest-neighbor to avoid blur
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(pixels.shape[1], pixels.shape[2]),
            mode="nearest"
        )

        # 2) Pad pixels and mask to multiples of 8 (no cropping, avoids shift)
        orig_pixels = pixels
        pixels = orig_pixels.clone()
        # Convert to NCHW for padding
        pixels_chw = pixels.permute(0, 3, 1, 2)
        _, _, H, W = pixels_chw.shape
        pad_h = (8 - (H % 8)) % 8
        pad_w = (8 - (W % 8)) % 8
        # pad = (left, right, top, bottom)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        pixels_chw = torch.nn.functional.pad(pixels_chw, pad, mode='reflect')
        mask = torch.nn.functional.pad(mask, pad, mode='nearest')
        # Back to BHWC
        pixels = pixels_chw.permute(0, 2, 3, 1)

        # 3) Prepare masked pixels for concat encoding
        m = (1.0 - mask.round()).squeeze(1)  # 1=keep, 0=inpaint
        for c in range(3):
            pixels[:, :, :, c] = (pixels[:, :, :, c] - 0.5) * m + 0.5

        # 4) Encode latents
        concat_latent = vae.encode(pixels)
        orig_latent   = vae.encode(orig_pixels)
        ref_latent    = reference_latents  # already a dict {"samples": tensor}

        # 5) Build conditioning dict
        c = node_helpers.conditioning_set_values(
            conditioning,
            {"concat_latent_image": concat_latent,
             "concat_mask": mask}
        )
        # 6) Append the face-reference latent so it's only used in the mask
        conditioning = self._append_reference_latent(c, ref_latent)

        # 7) Prepare output latent (orig_latent covers full image)
        out_latent = {"samples": orig_latent}
        if noise_mask:
            out_latent["noise_mask"] = mask

        return (conditioning, out_latent)

NODE_CLASS_MAPPINGS = {
    "Kontext Inpainting Conditioning": FluxKontextInpaintingConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kontext Inpainting Conditioning": "Kontext Inpainting Conditioning",
}
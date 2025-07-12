# Copyright 2025 ZenAI, Inc.
# Author: @Trgtuan_10, @vuongminh1907 (modified)
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
                "reference_latents": ("LATENT", ),  # ← new face‐latent port
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
        # 1) Align dims to multiples of 8
        h8 = (pixels.shape[1] // 8) * 8
        w8 = (pixels.shape[2] // 8) * 8

        # 2) Upsample mask to image size
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(pixels.shape[1], pixels.shape[2]),
            mode="bilinear"
        )

        # 3) Crop or pad if needed
        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != h8 or pixels.shape[2] != w8:
            y_off = (pixels.shape[1] - h8) // 2
            x_off = (pixels.shape[2] - w8) // 2
            pixels = pixels[:, y_off:y_off+h8, x_off:x_off+w8, :]
            mask   = mask[:, :, y_off:y_off+h8, x_off:x_off+w8]

        # 4) Prepare masked pixels for concat encoding
        m = (1.0 - mask.round()).squeeze(1)  # 1=keep, 0=inpaint
        for c in range(3):
            pixels[:,:,:,c] = (pixels[:,:,:,c] - 0.5) * m + 0.5

        # 5) Encode latents
        concat_latent = vae.encode(pixels)
        orig_latent   = vae.encode(orig_pixels)
        ref_latent    = reference_latents  # already a dict {"samples": tensor}

        # 6) Build conditioning dict
        c = node_helpers.conditioning_set_values(
            conditioning,
            {
                "concat_latent_image": concat_latent,
                "concat_mask": mask
            }
        )
        # 7) Append the face‐reference latent so it's only used in the mask
        conditioning = self._append_reference_latent(c, ref_latent)

        # 8) Prepare output latent
        out_latent = {"samples": orig_latent}
        if noise_mask:
            out_latent["noise_mask"] = mask

        return (conditioning, out_latent) 
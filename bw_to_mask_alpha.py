import torch


class BWToMaskAlpha:
    """Convert a black-and-white image into a mask / alpha (black = transparent)."""

    CATEGORY = "mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "rgba")
    FUNCTION = "convert"

    def convert(self, image, invert):
        # IMAGE: [B, H, W, C] in 0–1 → luminance as alpha/mask
        if image.shape[-1] >= 3:
            mask = (
                0.2126 * image[..., 0]
                + 0.7152 * image[..., 1]
                + 0.0722 * image[..., 2]
            )
        else:
            mask = image[..., 0]

        if invert:
            mask = 1.0 - mask

        mask = mask.clamp(0.0, 1.0)

        # RGBA for compositing: keep RGB, alpha = mask (black → transparent)
        rgb = image[..., :3] if image.shape[-1] >= 3 else image[..., :1].expand(-1, -1, -1, 3)
        rgba = torch.cat([rgb, mask.unsqueeze(-1)], dim=-1)

        return (mask, rgba)

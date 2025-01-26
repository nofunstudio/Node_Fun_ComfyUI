# __init__.py
from .layered_infinite_zoom import LayeredInfiniteZoom
from .replicate_API import APIGenerateReplicate

NODE_CLASS_MAPPINGS = {
    "LayeredInfiniteZoom": LayeredInfiniteZoom,
    "APIGenerateReplicate": APIGenerateReplicate
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayeredInfiniteZoom": "Layered Infinite Zoom",
    "APIGenerateReplicate": "Replicate API"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

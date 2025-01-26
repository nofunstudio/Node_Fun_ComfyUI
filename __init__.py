# __init__.py
from .layered_infinite_zoom import LayeredInfiniteZoom
from .replicate_API import ReplicateAPI_flux_1_1_pro_ultra

NODE_CLASS_MAPPINGS = {
    "LayeredInfiniteZoom": LayeredInfiniteZoom,
    "ReplicateAPI_flux_1_1_pro_ultra": ReplicateAPI_flux_1_1_pro_ultra
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayeredInfiniteZoom": "Layered Infinite Zoom",
    "ReplicateAPI_flux_1_1_pro_ultra": "Replicate flux 1.1 pro ultra"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

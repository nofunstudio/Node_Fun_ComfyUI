# __init__.py
from .layered_infinite_zoom import LayeredInfiniteZoom
from .replicate_Flux_Pro_Ultra import ReplicateAPI_flux_1_1_pro_ultra
from .replicate_Flux_Fill import ReplicateAPI_flux_fill_pro
from .iframe_view import IframeView

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "LayeredInfiniteZoom": LayeredInfiniteZoom,
    "ReplicateAPI_flux_1_1_pro_ultra": ReplicateAPI_flux_1_1_pro_ultra,
    "ReplicateAPI_flux_fill_pro": ReplicateAPI_flux_fill_pro,
    "IframeView": IframeView
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayeredInfiniteZoom": "Layered Infinite Zoom",
    "ReplicateAPI_flux_1_1_pro_ultra": "Replicate flux 1.1 pro ultra",
    "ReplicateAPI_flux_fill_pro": "Replicate flux fill pro",
    "IframeView": "Iframe View"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]

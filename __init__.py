# __init__.py
from .layered_infinite_zoom import LayeredInfiniteZoom
from .replicate_Flux_Pro_Ultra import ReplicateAPI_flux_1_1_pro_ultra
from .replicate_Flux_Fill import ReplicateAPI_flux_fill_pro
from .iframe_view import IframeView
from .queue_counter_reset import DynamicQueueCounter
from .custom_name_selector import IndexedStringSelector
from .ksampler_fun import KSampler
WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "LayeredInfiniteZoom": LayeredInfiniteZoom,
    "ReplicateAPI_flux_1_1_pro_ultra": ReplicateAPI_flux_1_1_pro_ultra,
    "ReplicateAPI_flux_fill_pro": ReplicateAPI_flux_fill_pro,
    "IframeView": IframeView,
    "DynamicQueueCounter": DynamicQueueCounter,
    "IndexedStringSelector": IndexedStringSelector,
    "Fun KSampler": KSampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayeredInfiniteZoom": "Layered Infinite Zoom",
    "ReplicateAPI_flux_1_1_pro_ultra": "Replicate flux 1.1 pro ultra",
    "ReplicateAPI_flux_fill_pro": "Replicate flux fill pro",
    "IframeView": "Iframe View",
    "DynamicQueueCounter": "Dynamic Queue Counter",
    "IndexedStringSelector": "Indexed String Selector",
    "Fun KSampler": "Fun KSampler"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]

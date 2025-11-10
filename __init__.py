# __init__.py
from .layered_infinite_zoom import LayeredInfiniteZoom
from .replicate_Flux_Pro_Ultra import ReplicateAPI_flux_1_1_pro_ultra
from .replicate_Flux_Fill import ReplicateAPI_flux_fill_pro
from .iframe_view import IframeView
from .queue_counter_reset import DynamicQueueCounter
from .custom_name_selector import IndexedStringSelector
from .ksampler_fun import KSampler
from .fal_recraft_upscale import FalAPI_recraft_upscale
from .fal_kling_video import FalAPI_kling_video
from .fal_seedance_video import FalAPI_seedance_video
from .wavespeed_ai_image import WaveSpeedAI_Image
from .string_lower import StringLower
from .multi_alpha_composite import MultiAlphaComposite
from .flux_kontext_inpaint_conditioning import FluxKontextInpaintingConditioning
from .load_image_batch_fun import LoadImageBatchFun
WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "LayeredInfiniteZoom": LayeredInfiniteZoom,
    "ReplicateAPI_flux_1_1_pro_ultra": ReplicateAPI_flux_1_1_pro_ultra,
    "ReplicateAPI_flux_fill_pro": ReplicateAPI_flux_fill_pro,
    "IframeView": IframeView,
    "DynamicQueueCounter": DynamicQueueCounter,
    "IndexedStringSelector": IndexedStringSelector,
    "Fun KSampler": KSampler,
    "FalAPI_recraft_upscale": FalAPI_recraft_upscale,
    "FalAPI_kling_video": FalAPI_kling_video,
    "FalAPI_seedance_video": FalAPI_seedance_video,
    "WaveSpeedAI_Image": WaveSpeedAI_Image,
    "StringLower": StringLower,
    "MultiAlphaComposite": MultiAlphaComposite,
    "FluxKontextInpaintingConditioning": FluxKontextInpaintingConditioning,
    "LoadImageBatchFun": LoadImageBatchFun
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayeredInfiniteZoom": "Layered Infinite Zoom",
    "ReplicateAPI_flux_1_1_pro_ultra": "Replicate flux 1.1 pro ultra",
    "ReplicateAPI_flux_fill_pro": "Replicate flux fill pro",
    "IframeView": "Iframe View",
    "DynamicQueueCounter": "Dynamic Queue Counter",
    "IndexedStringSelector": "Indexed String Selector",
    "Fun KSampler": "Fun KSampler",
    "FalAPI_recraft_upscale": "Fal API Recraft Upscale",
    "FalAPI_kling_video": "Fal API Kling Video",
    "FalAPI_seedance_video": "Fal API Seedance Video",
    "WaveSpeedAI_Image": "WaveSpeedAI Image Generation",
    "StringLower": "String to Lowercase",
    "MultiAlphaComposite": "Multi Alpha Composite",
    "FluxKontextInpaintingConditioning": "No Fun Flux Kontext Inpaint Conditioning",
    "LoadImageBatchFun": "Load Image Batch (Fun)"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]

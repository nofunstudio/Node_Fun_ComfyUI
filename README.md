# Node_Fun_ComfyUI

A collection of custom nodes for ComfyUI featuring API integrations, image processing utilities, and creative tools for generative workflows.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-repo/Node_Fun_ComfyUI.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI

---

## Nodes

### 🖼️ Image Processing

#### **Layered Infinite Zoom 2x**
Creates a smooth infinite zoom animation by layering 5 images on top of each other, scaling from 200% down to smaller sizes. Perfect for creating hypnotic zoom-in video effects with optional ease-in-out-cubic smoothing.

![layeredZoom](https://github.com/user-attachments/assets/0bf5c7e5-6023-44b7-88d5-e00e58f61ce5)

#### **Multi Alpha Composite**
Composites up to 8 images together using alpha blending, similar to Photoshop's layer stacking. Images are layered bottom-to-top, with transparent regions allowing lower layers to show through.

#### **Load Image Batch (Fun)**
A fixed version of Load Image Batch that properly supports multiple instances in the same workflow. Each instance maintains its own state, preventing conflicts when you need several batch loaders simultaneously. Supports single, incremental, or random image selection modes.

---

### 🎬 3D Viewer & Iframe

#### **Iframe View**
Embeds a 3D scene viewer directly into ComfyUI by loading any URL in an iframe. Built for R3F (React Three Fiber) and Three.js applications, this node captures **four texture outputs**:
- **Color** – The rendered scene
- **Depth** – Distance-based depth map
- **Normal** – Surface normal vectors
- **Canny** – Edge detection

Use this to view 3D models, capture multi-pass renders for ControlNet conditioning, or integrate external web-based 3D tools into your workflow. Supports animation frame capture with configurable frame counts and scene state control via JSON.

---

### 🤖 FAL API Integrations

These nodes require a [FAL API key](https://fal.ai). Set your key in the node or via the `FAL_KEY` environment variable.

#### **FAL Qwen Edit Plus**
Image editing powered by the Qwen model. Takes up to 3 input images and a text prompt to generate edited results. Supports custom LoRA models for style transfer and fine-tuned edits.

#### **FAL Flux 2 LoRA Edit**
Edit images using Black Forest Labs' Flux 2 model with optional LoRA support. Accepts up to 3 reference images, with controls for guidance scale, inference steps, and prompt expansion.

#### **FAL Flux 2 Turbo Edit**
A faster, streamlined version of Flux 2 editing optimized for quick iterations. Fewer parameters for simpler workflows when speed matters more than fine-tuning.

#### **FAL Flux 2 Klein Edit (4B/9B)**
Image editing using the Flux 2 Klein models. Toggle between the 4B (faster) and 9B (higher quality) model variants. Accepts up to 3 reference images with optional LoRA support and negative prompts. Empty/black images on inputs 2 and 3 are automatically ignored.

#### **FAL Kling Video Generation**
Generates video from a single image using Kling Video v2.1. Provide an image and a motion prompt to create AI-generated video clips. Output is saved as MP4.

#### **FAL Seedance Video Generation**
Video generation using ByteDance's Seedance v1.5 Pro model. Features include:
- Configurable aspect ratio (21:9 to 9:16)
- Resolution options (480p/720p)
- Duration control (3-12 seconds)
- Optional end image for transitions
- Auto-generated audio
- Fixed or moving camera modes

#### **FAL Recraft Crisp Upscale**
High-quality image upscaling using Recraft's crisp upscale model. Enhances image resolution while preserving sharp details and textures.

#### **Nano Banana Pro (FAL)**
Multi-image generation/editing via Nano Banana Pro. Takes up to 4 input images and generates new images at 1K, 2K, or 4K resolution with various aspect ratios.

---

### 🔄 Replicate API Integrations

These nodes require a [Replicate API token](https://replicate.com).

#### **Replicate Flux-Fill-Pro**
Outpainting and inpainting using Flux Fill Pro. Provide an image, optional mask, and an outpaint direction (e.g., "Zoom out 2x") to extend or fill image regions.

#### **Replicate Flux 1.1 Pro Ultra**
High-quality image generation using Flux 1.1 Pro Ultra. Supports batch generation of multiple images and optional image prompts for image-to-image guidance with adjustable prompt strength.

---

### ⚡ WaveSpeedAI Integrations

These nodes require a [WaveSpeedAI API key](https://wavespeed.ai).

#### **WaveSpeedAI Image Generation**
Image generation using WaveSpeedAI's Qwen image editing endpoint. Takes up to 3 input images with optional LoRA support for style control.

#### **WaveSpeedAI Flux 2 LoRA Edit**
Edit images using WaveSpeedAI's Flux 2 model. Similar to the FAL version but using WaveSpeedAI's infrastructure.

---

### 🔧 Sampling & Conditioning

#### **Fun KSampler**
A KSampler variant that saves intermediate preview images during the sampling process. Uses TAESD for fast, high-quality previews saved directly to your output folder. Configure how many preview steps to capture and how many initial steps to skip.

#### **Kontext Inpainting Conditioning**
Prepares conditioning for Flux Kontext inpainting workflows. Combines VAE-encoded pixels, masks, and reference latents (e.g., face latents) for context-aware inpainting that respects reference features.

---

### 🛠️ Utilities

#### **Indexed String Selector**
Selects a single string from a comma-separated list by index. Useful for iterating through style names, prompts, or any list of text values in batch workflows.

#### **Dynamic Queue Counter**
A persistent counter that increments or decrements across queue runs. Set start, stop, and step values to create loops, and use the reset toggle to restart the sequence. Returns float, int, and info string outputs.

#### **String Lower**
Simple utility that converts any string to lowercase. Useful for normalizing text inputs.

---

## API Key Setup

For API-based nodes, you can provide keys in three ways:
1. **Node input** – Paste directly into the node's API key field
2. **Environment variable** – Set `FAL_KEY`, `REPLICATE_API_TOKEN`, or `WAVESPEED_API_KEY`
3. **Config file** – Some nodes check for keys in standard config locations

---

## Output Locations

Generated images and videos are saved to:
- **FAL nodes**: `output/API/FAL/{model-name}/`
- **Replicate nodes**: `output/API/Replicate/`
- **WaveSpeedAI nodes**: ComfyUI temp directory
- **Fun KSampler previews**: `output/` folder

Metadata JSON files are saved alongside outputs in a `metadata/` subfolder.

---

## Contributing

Issues and pull requests are welcome! Please ensure any new nodes follow the existing patterns for error handling and output saving.

## License

MIT License

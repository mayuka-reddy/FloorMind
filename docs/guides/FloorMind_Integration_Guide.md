
# FloorMind Model Integration Guide

## Method 1: Using Diffusers Pipeline (Recommended)
```python
from diffusers import StableDiffusionPipeline
import torch

# Load the pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "path/to/your/google/folder",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
)

# Move to appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)

# Generate floor plan
prompt = "Modern 3-bedroom apartment floor plan with open kitchen"
image = pipeline(
    prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

image.save("generated_floor_plan.png")
```

## Method 2: Manual Component Loading
```python
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors.torch
import torch

# Load base components
base_model = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

# Load fine-tuned UNet
unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
state_dict = safetensors.torch.load_file("google/model.safetensors")
unet.load_state_dict(state_dict)

# Create pipeline
pipeline = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None
)
```

## Method 3: Backend Integration
```python
class FloorMindGenerator:
    def __init__(self, model_path="google"):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = self.pipeline.to(self.device)
    
    def generate_floor_plan(self, description, **kwargs):
        return self.pipeline(
            description,
            num_inference_steps=kwargs.get('steps', 20),
            guidance_scale=kwargs.get('guidance', 7.5),
            height=kwargs.get('height', 512),
            width=kwargs.get('width', 512)
        ).images[0]
```

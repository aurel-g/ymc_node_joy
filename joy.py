import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoModel, 
    AutoProcessor, 
    AutoTokenizer, 
    PreTrainedTokenizer, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig,
    AutoModelForCausalLM
    
)
import folder_paths
from model_management import get_torch_device

from .lib.ximg import tensor2pil, pil2tensor
from .lib.xmodel import download_hg_model
from .conf import CURRENT_CATEGORY, CURRENT_FUNCTION

DEVICE = get_torch_device()

class ImageAdapter(nn.Module):
    """Adapter to transform image features to match text model hidden size."""
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class JoyPipeline:
    """Pipeline for handling image captioning models."""
    def __init__(self):
        self.clip_model: Optional[nn.Module] = None
        self.clip_processor: Optional[AutoProcessor] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.text_model: Optional[nn.Module] = None
        self.image_adapter: Optional[nn.Module] = None
        self.parent: Optional[object] = None
    
    def clear_cache(self) -> None:
        """Clear all model components from memory."""
        for attr in ['clip_model', 'clip_processor', 'tokenizer', 'text_model', 'image_adapter']:
            setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class JoyCaptionBase:
    """Base class for Joy captioning functionality."""
    def __init__(self):
        self.model: Optional[str] = None
        self.pipeline = JoyPipeline()
        self.pipeline.parent = self

    def load_checkpoint(self, model_id: str) -> None:
        """Load all required models and components."""

        print(f"Loading model: {model_id}")
        if self.pipeline.clip_model is not None and self.model == model_id:
            print("Model already loaded, skipping")
            return
            
        print("Loading CLIP model...")
        self.pipeline.clear_cache()
        self.model = model_id
        
        # Load CLIP model
        clip_model_id = "google/siglip-so400m-patch14-384"
        clip_path = download_hg_model(clip_model_id, "clip")
        
        self.pipeline.clip_processor = AutoProcessor.from_pretrained(clip_path)
        clip_model = AutoModel.from_pretrained(
            clip_path,
            trust_remote_code=True
        ).vision_model
        
        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_model.to(DEVICE)
        self.pipeline.clip_model = clip_model

        # Load LLM
        model_path = download_hg_model(model_id, "LLM")
        self.pipeline.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=False
        )
        
        # todo: fix Unused kwargs: ['_load_in_4bit', '_load_in_8bit', 'quant_method']
        bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=True,
          bnb_4bit_compute_dtype=torch.float16
        )
        self.pipeline.text_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            max_memory={0: "10GiB"}
        )
    
        self.pipeline.text_model.eval()

        # Load image adapter
        adapter_path = os.path.join(
            folder_paths.models_dir, 
            "Joy_caption", 
            "image_adapter.pt"
        )
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Image adapter not found at {adapter_path}")
            
        image_adapter = ImageAdapter(
            clip_model.config.hidden_size,
            self.pipeline.text_model.config.hidden_size
        )
        image_adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
        image_adapter.eval()
        image_adapter.to(DEVICE)
        self.pipeline.image_adapter = image_adapter

    def generate_caption(
        self,
        image: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate caption for a single image."""
        if self.pipeline.clip_processor is None:
            raise RuntimeError("Pipeline not initialized. Call load_checkpoint first.")
            
        # Convert and preprocess image
        input_image = tensor2pil(image)
        p_image = self.pipeline.clip_processor(
            images=input_image, 
            return_tensors='pt'
        ).pixel_values.to(DEVICE)

        # Tokenize prompt
        prompt_tokens = self.pipeline.tokenizer.encode(
            prompt, 
            return_tensors='pt', 
            add_special_tokens=False
        ).to(DEVICE)

        with torch.autocast(device_type=str(DEVICE), enabled=True):
            # Get image features
            vision_outputs = self.pipeline.clip_model(
                pixel_values=p_image, 
                output_hidden_states=True
            )
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = self.pipeline.image_adapter(image_features)

            # Prepare embeddings
            prompt_embeds = self.pipeline.text_model.model.embed_tokens(prompt_tokens)
            embedded_bos = self.pipeline.text_model.model.embed_tokens(
                torch.tensor(
                    [[self.pipeline.tokenizer.bos_token_id]], 
                    device=DEVICE
                )
            )

            # Construct input embeddings
            inputs_embeds = torch.cat([
                embedded_bos.expand(embedded_images.shape[0], -1, -1),
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(embedded_images.shape[0], -1, -1),
            ], dim=1)

            # Generate caption
            generate_ids = self.pipeline.text_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=10,
                temperature=temperature
            )

            # Decode and clean caption
            caption = self.pipeline.tokenizer.decode(
                generate_ids[0], 
                skip_special_tokens=True
            ).strip()
            
        return caption

class JoyCaptionLoad(JoyCaptionBase):
    """Node to load Joy captioning models."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "unsloth/Meta-Llama-3.1-8B-bnb-4bit", 
                    "meta-llama/Meta-Llama-3.1-8B"
                ],), 
            }
        }

    CATEGORY = CURRENT_CATEGORY
    # FUNCTION = CURRENT_FUNCTION
    NODE_DESC = "joy model loader"
    RETURN_TYPES = ("JoyPipeline",)
    FUNCTION = "load"
    OUTPUT_NODE = True
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    def load(self, model: str) -> Tuple[JoyPipeline]:
        self.load_checkpoint(model)
        return (self.pipeline,)

class JoyCaption(JoyCaptionBase):
    """Node to generate captions for single images."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joy_pipeline": ("JoyPipeline",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A descriptive caption for this image"
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024, 
                    "min": 10, 
                    "max": 4096, 
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "cache": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = CURRENT_CATEGORY
    # FUNCTION = CURRENT_FUNCTION

    NODE_DESC = "joy image caption"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    def generate(
        self,
        joy_pipeline: JoyPipeline,
        image: torch.Tensor,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        cache: bool
    ) -> Tuple[str]:
        try:
            self.pipeline = joy_pipeline
            caption = self.generate_caption(image, prompt, max_new_tokens, temperature)
            
            if not cache:
                self.pipeline.clear_cache()
            
            # ensure caption to be in oneline
            caption = ' '.join(caption.split())
            return (caption,)
        except Exception as e:
            raise RuntimeError(f"Caption generation failed: {str(e)}")
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class JoyCaptionFromDir(JoyCaptionBase):
    """Node to generate captions for all images in a directory."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joy_pipeline": ("JoyPipeline",),
                "image_dir": ("STRING", {
                    "default": "", 
                    "multiline": False
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A descriptive caption for these images"
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024, 
                    "min": 10, 
                    "max": 4096, 
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "cache": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = CURRENT_CATEGORY
    # FUNCTION = CURRENT_FUNCTION
    NODE_DESC = "joy diretory caption"
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_files", "captions")
    FUNCTION = "generate_for_dir"

    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    def generate_for_dir(
        self,
        joy_pipeline: JoyPipeline,
        image_dir: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        cache: bool
    ) -> Tuple[str, str]:

        if not os.path.isdir(image_dir):
            raise ValueError(f"Directory not found: {image_dir}")
            
        # Get all image files from directory
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(image_extensions)]
                  
        if not image_files:
            print(f"No images found in directory: {image_dir}")
            return ('','',)

  
        self.pipeline = joy_pipeline
        files = []
        captions = []
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            try:
                image = Image.open(img_path)
                tensor_image = pil2tensor(image)
                caption = self.generate_caption(
                    tensor_image, 
                    prompt, 
                    max_new_tokens, 
                    temperature
                )
                # ensure caption to be in oneline
                caption = ' '.join(caption.split())

                files.append(img_path)
                captions.append(caption)
                print(f"Processed: {img_file}")
            except Exception as e:
                # captions.append(f"{img_file}: Error - {str(e)}")
                print(f"{img_file}: Error - {str(e)}")
                continue
        if not cache:
            self.pipeline.clear_cache()
            
        return (
            "\n".join(files), 
            "\n".join(captions)
        )

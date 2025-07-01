import os
import torch
from diffusers import StableDiffusionPipeline
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages Stable Diffusion models and LoRA weights"""
    
    def __init__(self):
        self.base_model_cache = {}
        self.lora_models = {}
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_base_model(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Load base Stable Diffusion model"""
        try:
            if model_name not in self.base_model_cache:
                logger.info(f"Loading base model: {model_name}")
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                pipeline = pipeline.to(self.device)
                self.base_model_cache[model_name] = pipeline
                logger.info(f"Base model {model_name} loaded successfully")
            
            return self.base_model_cache[model_name]
        except Exception as e:
            logger.error(f"Error loading base model {model_name}: {str(e)}")
            raise
    
    def load_lora_model(self, model_name: str, lora_path: str):
        """Load LoRA weights for fine-tuned model"""
        try:
            base_model = self.load_base_model()
            # This would load LoRA weights - simplified for now
            logger.info(f"Loading LoRA model: {model_name} from {lora_path}")
            # TODO: Implement actual LoRA loading
            self.lora_models[model_name] = {
                "base_model": base_model,
                "lora_path": lora_path
            }
            return base_model
        except Exception as e:
            logger.error(f"Error loading LoRA model {model_name}: {str(e)}")
            raise
    
    def list_available_models(self) -> List[str]:
        """List all available models (base + LoRA)"""
        models = list(self.base_model_cache.keys())
        models.extend(self.lora_models.keys())
        return models
    
    def generate_image(self, 
                      prompt: str,
                      model_name: Optional[str] = None,
                      negative_prompt: str = "",
                      num_inference_steps: int = 20,
                      guidance_scale: float = 7.5,
                      height: int = 512,
                      width: int = 512,
                      num_images: int = 1,
                      seed: Optional[int] = None) -> List[Any]:
        """Generate images using specified model"""
        try:
            # Use LoRA model if available, otherwise base model
            if model_name and model_name in self.lora_models:
                pipeline = self.lora_models[model_name]["base_model"]
            else:
                pipeline = self.load_base_model()
            
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate images
            logger.info(f"Generating {num_images} image(s) with prompt: {prompt[:50]}...")
            
            # Use appropriate autocast context for device
            if self.device == "mps":
                # MPS doesn't support autocast in the same way
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images
                )
            else:
                with torch.autocast(self.device):
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        num_images_per_prompt=num_images
                    )
            
            return result.images
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from cache"""
        try:
            if model_name in self.lora_models:
                del self.lora_models[model_name]
                logger.info(f"LoRA model {model_name} deleted")
                return True
            elif model_name in self.base_model_cache:
                del self.base_model_cache[model_name]
                logger.info(f"Base model {model_name} deleted from cache")
                return True
            else:
                logger.warning(f"Model {model_name} not found")
                return False
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {str(e)}")
            return False 
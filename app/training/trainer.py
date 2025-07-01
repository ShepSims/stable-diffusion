import os
import torch
import asyncio
from pathlib import Path
from typing import Optional, Callable
import logging
from datetime import datetime

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from datasets import Dataset

from ..config import settings

logger = get_logger(__name__)

class LoRATrainer:
    def __init__(self):
        self.accelerator = Accelerator(
            mixed_precision=settings.MIXED_PRECISION,
            gradient_accumulation_steps=1,
            log_with="tensorboard",
            project_dir="logs"
        )
        
    async def train_lora(
        self,
        job_id: str,
        model_name: str,
        training_images_dir: str,
        training_steps: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        resolution: int = 512,
        use_8bit_adam: bool = True,
        gradient_checkpointing: bool = True,
        mixed_precision: str = "fp16",
        progress_callback: Optional[Callable] = None
    ):
        """Train LoRA adapters for Stable Diffusion"""
        
        logger.info(f"Starting LoRA training for job {job_id}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Load the base model
        pipeline = StableDiffusionPipeline.from_pretrained(
            settings.DEFAULT_MODEL,
            torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device
        pipeline = pipeline.to(self.accelerator.device)
        
        # Enable gradient checkpointing
        if gradient_checkpointing:
            pipeline.unet.enable_gradient_checkpointing()
            pipeline.text_encoder.gradient_checkpointing_enable()
        
        # Set up LoRA
        unet = pipeline.unet
        text_encoder = pipeline.text_encoder
        
        # Add LoRA adapters
        unet_lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            unet_lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=4,
            )
        
        unet.set_attn_processor(unet_lora_attn_procs)
        
        # Prepare dataset
        dataset = self._prepare_dataset(training_images_dir, resolution)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer
        lora_layers = AttnProcsLayers(unet.attn_processors)
        
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(lora_layers.parameters(), lr=learning_rate)
            except ImportError:
                logger.warning("bitsandbytes not available, using regular AdamW")
                optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=learning_rate)
        
        # Prepare with accelerator
        lora_layers, optimizer, dataloader = self.accelerator.prepare(
            lora_layers, optimizer, dataloader
        )
        
        # Training loop
        global_step = 0
        
        for epoch in range((training_steps // len(dataloader)) + 1):
            for batch in dataloader:
                if global_step >= training_steps:
                    break
                
                with self.accelerator.accumulate(lora_layers):
                    # Convert images to latent space
                    latents = pipeline.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * pipeline.vae.config.scaling_factor
                    
                    # Sample noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    # Sample random timestep
                    timesteps = torch.randint(
                        0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()
                    
                    # Add noise to latents
                    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get text embeddings
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    
                    # Predict noise
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    
                    # Compute loss
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(global_step, training_steps)
                
                if global_step % 100 == 0:
                    logger.info(f"Step {global_step}/{training_steps}, Loss: {loss.item():.4f}")
                
                if global_step >= training_steps:
                    break
        
        # Save LoRA weights
        model_save_path = os.path.join(settings.MODELS_DIR, model_name)
        os.makedirs(model_save_path, exist_ok=True)
        
        # Save LoRA weights
        lora_state_dict = {
            f"unet.{module_name}": param
            for module_name, param in lora_layers.state_dict().items()
        }
        
        torch.save(lora_state_dict, os.path.join(model_save_path, "pytorch_lora_weights.bin"))
        
        # Save training config
        config = {
            "base_model": settings.DEFAULT_MODEL,
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "resolution": resolution,
            "timestamp": datetime.now().isoformat(),
            "job_id": job_id
        }
        
        import json
        with open(os.path.join(model_save_path, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training completed for job {job_id}")
        
    def _prepare_dataset(self, images_dir: str, resolution: int):
        """Prepare dataset from training images"""
        
        # Get image paths
        image_paths = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_paths.extend(Path(images_dir).glob(f"*{ext}"))
        
        # Define transforms
        train_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Create dataset
        dataset_dict = {
            "pixel_values": [],
            "input_ids": []
        }
        
        tokenizer = CLIPTokenizer.from_pretrained(settings.DEFAULT_MODEL, subfolder="tokenizer")
        
        for image_path in image_paths:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            image = train_transforms(image)
            dataset_dict["pixel_values"].append(image)
            
            # Simple prompt - can be enhanced with actual captions
            prompt = "a photo in the style"
            input_ids = tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids[0]
            
            dataset_dict["input_ids"].append(input_ids)
        
        return Dataset.from_dict(dataset_dict)
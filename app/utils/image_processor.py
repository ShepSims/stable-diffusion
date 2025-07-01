import os
import asyncio
from typing import List, Tuple
from pathlib import Path
import logging

from PIL import Image, ImageOps
import numpy as np
import cv2

from ..config import settings

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
        self.target_resolution = settings.DEFAULT_RESOLUTION
        
    async def process_training_image(self, image_path: str) -> str:
        """Process and validate a single training image"""
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get original dimensions
            original_width, original_height = image.size
            
            # Validate minimum size
            min_dimension = min(original_width, original_height)
            if min_dimension < 256:
                raise ValueError(f"Image too small: {min_dimension}x{min_dimension}. Minimum size is 256x256")
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > settings.MAX_FILE_SIZE:
                raise ValueError(f"File size {file_size} exceeds maximum allowed size {settings.MAX_FILE_SIZE}")
            
            # Process image for training
            processed_image = await self._preprocess_for_training(image)
            
            # Save processed image
            processed_path = image_path.replace(Path(image_path).suffix, "_processed.png")
            processed_image.save(processed_path, "PNG", quality=95)
            
            logger.info(f"Processed image: {image_path} -> {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise
    
    async def _preprocess_for_training(self, image: Image.Image) -> Image.Image:
        """Preprocess image for training"""
        
        # Get image dimensions
        width, height = image.size
        
        # Calculate target size maintaining aspect ratio
        if width > height:
            new_width = self.target_resolution
            new_height = int((height * self.target_resolution) / width)
        else:
            new_height = self.target_resolution
            new_width = int((width * self.target_resolution) / height)
        
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop to target resolution
        image = ImageOps.fit(image, (self.target_resolution, self.target_resolution), Image.Resampling.LANCZOS)
        
        # Enhance image quality
        image = await self._enhance_image(image)
        
        return image
    
    async def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements"""
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img_array, -1, kernel)
        
        # Blend original and sharpened (subtle effect)
        enhanced = cv2.addWeighted(img_array, 0.8, sharpened, 0.2, 0)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(enhanced)
        
        return enhanced_image
    
    async def batch_process_images(self, image_paths: List[str]) -> List[str]:
        """Process multiple images concurrently"""
        
        tasks = []
        for image_path in image_paths:
            task = asyncio.create_task(self.process_training_image(image_path))
            tasks.append(task)
        
        processed_paths = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful paths
        successful_paths = []
        for i, result in enumerate(processed_paths):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {image_paths[i]}: {result}")
            else:
                successful_paths.append(result)
        
        return successful_paths
    
    def validate_image_format(self, file_path: str) -> bool:
        """Validate if file is a supported image format"""
        
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_formats
    
    def get_image_info(self, image_path: str) -> dict:
        """Get image information"""
        
        try:
            with Image.open(image_path) as img:
                return {
                    "filename": Path(image_path).name,
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "file_size": os.path.getsize(image_path)
                }
        except Exception as e:
            return {"error": str(e)}
    
    async def create_preview_grid(self, image_paths: List[str], output_path: str) -> str:
        """Create a grid preview of training images"""
        
        if not image_paths:
            raise ValueError("No image paths provided")
        
        # Limit to first 25 images
        image_paths = image_paths[:25]
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(len(image_paths))))
        
        # Load and resize images
        images = []
        preview_size = 128
        
        for path in image_paths:
            try:
                img = Image.open(path)
                img = img.convert('RGB')
                img = ImageOps.fit(img, (preview_size, preview_size), Image.Resampling.LANCZOS)
                images.append(img)
            except:
                # Create placeholder for failed images
                placeholder = Image.new('RGB', (preview_size, preview_size), color='gray')
                images.append(placeholder)
        
        # Create grid image
        grid_width = grid_size * preview_size
        grid_height = grid_size * preview_size
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Paste images into grid
        for i, img in enumerate(images):
            row = i // grid_size
            col = i % grid_size
            x = col * preview_size
            y = row * preview_size
            grid_image.paste(img, (x, y))
        
        # Save grid
        grid_image.save(output_path, "PNG")
        
        return output_path
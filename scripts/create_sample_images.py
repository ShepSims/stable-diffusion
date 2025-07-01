#!/usr/bin/env python3
"""
Create sample training images for testing the API
"""

import os
from PIL import Image, ImageDraw, ImageFont
import random
import argparse
from pathlib import Path

def create_sample_image(width=512, height=512, style="abstract"):
    """Create a sample image with given style"""
    
    # Create a new image with random background color
    bg_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    if style == "abstract":
        # Draw some abstract shapes
        for _ in range(random.randint(3, 8)):
            shape_type = random.choice(['rectangle', 'ellipse', 'line'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            if shape_type == 'rectangle':
                x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
                x2, y2 = random.randint(x1, width), random.randint(y1, height)
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
            
            elif shape_type == 'ellipse':
                x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
                x2, y2 = random.randint(x1, width), random.randint(y1, height)
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
            
            elif shape_type == 'line':
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = random.randint(0, width), random.randint(0, height)
                draw.line([x1, y1, x2, y2], fill=color, width=random.randint(2, 10))
    
    elif style == "geometric":
        # Draw geometric patterns
        for _ in range(random.randint(5, 15)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Draw triangles
            points = []
            for _ in range(3):
                points.append((random.randint(0, width), random.randint(0, height)))
            draw.polygon(points, fill=color)
    
    elif style == "gradient":
        # Create gradient effect
        for y in range(height):
            for x in range(width):
                r = int(255 * (x / width))
                g = int(255 * (y / height))
                b = int(255 * ((x + y) / (width + height)))
                draw.point((x, y), fill=(r, g, b))
    
    elif style == "textured":
        # Create textured pattern
        for _ in range(random.randint(50, 200)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(1, 5)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
    
    return image

def create_sample_dataset(output_dir, num_images=25, style="mixed"):
    """Create a dataset of sample images"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    styles = ["abstract", "geometric", "gradient", "textured"]
    
    print(f"Creating {num_images} sample images in {output_dir}")
    
    for i in range(num_images):
        if style == "mixed":
            current_style = random.choice(styles)
        else:
            current_style = style
        
        # Create image
        image = create_sample_image(style=current_style)
        
        # Save image
        filename = f"sample_{i:03d}_{current_style}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath, "PNG")
        
        print(f"Created: {filename}")
    
    print(f"\n✅ Created {num_images} sample images in {output_dir}")
    print(f"You can now test the API with: python scripts/test_api.py --image-dir {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create sample training images")
    parser.add_argument("--output-dir", default="sample_images", help="Output directory for images")
    parser.add_argument("--num-images", type=int, default=25, help="Number of images to create")
    parser.add_argument("--style", choices=["abstract", "geometric", "gradient", "textured", "mixed"], 
                       default="mixed", help="Style of images to create")
    parser.add_argument("--size", type=int, default=512, help="Image size (square)")
    
    args = parser.parse_args()
    
    create_sample_dataset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        style=args.style
    )

if __name__ == "__main__":
    main()
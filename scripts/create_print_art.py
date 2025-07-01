#!/usr/bin/env python3
"""
Create high-quality print-ready artwork using procedural generation
Perfect for wall art, posters, and fine art prints!
"""

import os
import math
import random
import colorsys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

# Print size presets (width, height) at 300 DPI
PRINT_SIZES = {
    "8x10": (2400, 3000),      # 8"x10" at 300 DPI
    "11x14": (3300, 4200),     # 11"x14" at 300 DPI  
    "16x20": (4800, 6000),     # 16"x20" at 300 DPI
    "24x36": (7200, 10800),    # 24"x36" poster at 300 DPI
    "square_12": (3600, 3600), # 12"x12" square at 300 DPI
    "square_16": (4800, 4800), # 16"x16" square at 300 DPI
    "square_24": (7200, 7200), # 24"x24" large square at 300 DPI
}

def extract_colors_from_image(image_path, num_colors=5, resize_to=150):
    """Extract dominant colors from an image using k-means clustering"""
    try:
        # Open and resize image for faster processing
        image = Image.open(image_path).convert('RGB')
        image = image.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        img_array = img_array.reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        try:
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(img_array)
            colors = kmeans.cluster_centers_.astype(int)
        except ImportError:
            # Fallback to histogram-based method if sklearn not available
            colors = extract_colors_histogram(img_array, num_colors)
        
        # Convert to tuples and ensure valid RGB values
        color_palette = []
        for color in colors:
            r, g, b = [max(0, min(255, int(c))) for c in color]
            color_palette.append((r, g, b))
        
        return color_palette
        
    except Exception as e:
        print(f"Error extracting colors from {image_path}: {e}")
        return get_color_palette("warm")  # Fallback to default palette

def extract_colors_histogram(img_array, num_colors=5):
    """Fallback method using color histogram (no sklearn required)"""
    # Reduce color space for better clustering
    img_array = (img_array // 32) * 32  # Quantize to reduce color space
    
    # Count color frequencies
    unique_colors, counts = np.unique(img_array.reshape(-1, 3), axis=0, return_counts=True)
    
    # Sort by frequency and take top colors
    sorted_indices = np.argsort(counts)[::-1]
    top_colors = unique_colors[sorted_indices[:num_colors]]
    
    return top_colors

def enhance_color_palette(colors, enhance_saturation=True, enhance_contrast=True):
    """Enhance extracted colors for better artistic results"""
    enhanced_colors = []
    
    for r, g, b in colors:
        # Convert to HSV for easier manipulation
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        
        if enhance_saturation:
            # Boost saturation for more vivid colors
            s = min(1.0, s * 1.3)
        
        if enhance_contrast:
            # Enhance contrast by pulling values toward extremes
            if v < 0.5:
                v = max(0.1, v * 0.8)  # Make dark colors darker
            else:
                v = min(1.0, v * 1.1)  # Make light colors lighter
        
        # Convert back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        enhanced_colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return enhanced_colors

def create_palette_preview(colors, output_path="palette_preview.png", size=(800, 200)):
    """Create a visual preview of the extracted color palette"""
    image = Image.new('RGB', size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    color_width = size[0] // len(colors)
    
    for i, color in enumerate(colors):
        x1 = i * color_width
        x2 = (i + 1) * color_width
        draw.rectangle([x1, 0, x2, size[1]], fill=color)
        
        # Add color hex value
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        draw.text((x1 + 10, size[1] - 30), hex_color, fill=(255, 255, 255) if sum(color) < 400 else (0, 0, 0))
    
    image.save(output_path)
    return image

def get_color_palette(palette_type="warm"):
    """Generate sophisticated color palettes for fine art"""
    
    if palette_type == "warm":
        # Warm sunset palette
        return [
            (255, 94, 77),   # Coral
            (255, 154, 0),   # Orange
            (255, 206, 84),  # Yellow
            (255, 118, 117), # Pink
            (240, 84, 84),   # Red
        ]
    elif palette_type == "cool":
        # Cool ocean palette
        return [
            (74, 144, 226),  # Blue
            (80, 227, 194),  # Teal
            (54, 215, 183),  # Cyan
            (129, 236, 236), # Light cyan
            (99, 110, 250),  # Purple-blue
        ]
    elif palette_type == "earth":
        # Earth tones
        return [
            (101, 67, 33),   # Brown
            (144, 102, 72),  # Tan
            (62, 95, 44),    # Forest green
            (139, 115, 85),  # Beige
            (160, 133, 101), # Light brown
        ]
    elif palette_type == "monochrome":
        # Sophisticated grays
        return [
            (45, 45, 45),    # Charcoal
            (95, 95, 95),    # Medium gray
            (145, 145, 145), # Light gray
            (195, 195, 195), # Very light gray
            (220, 220, 220), # Almost white
        ]
    elif palette_type == "jewel":
        # Rich jewel tones
        return [
            (183, 28, 28),   # Ruby
            (26, 35, 126),   # Sapphire
            (27, 94, 32),    # Emerald
            (69, 39, 160),   # Amethyst
            (230, 126, 34),  # Topaz
        ]

def create_organic_blob(draw, center, max_radius, color, complexity=12):
    """Create organic, flowing blob shapes"""
    cx, cy = center
    points = []
    
    for i in range(complexity):
        angle = (2 * math.pi * i) / complexity
        # Add some randomness to radius for organic feel
        radius = max_radius * (0.5 + 0.5 * random.random())
        # Add some noise to angle
        noise_angle = angle + (random.random() - 0.5) * 0.5
        
        x = cx + radius * math.cos(noise_angle)
        y = cy + radius * math.sin(noise_angle)
        points.append((x, y))
    
    draw.polygon(points, fill=color)

def create_mandala_pattern(image, center, max_radius, colors, symmetry=8):
    """Create beautiful mandala-style patterns"""
    draw = ImageDraw.Draw(image)
    cx, cy = center
    
    # Create multiple rings
    for ring in range(5):
        ring_radius = max_radius * (0.2 + 0.6 * ring / 4)
        
        for i in range(symmetry):
            angle = (2 * math.pi * i) / symmetry
            
            # Create petal-like shapes
            for j in range(3):
                petal_angle = angle + (j - 1) * 0.3
                petal_radius = ring_radius * (0.3 + 0.4 * random.random())
                
                x = cx + petal_radius * math.cos(petal_angle)
                y = cy + petal_radius * math.sin(petal_angle)
                
                blob_radius = petal_radius * 0.3
                color = random.choice(colors)
                
                create_organic_blob(draw, (x, y), blob_radius, color)

def create_flow_field(width, height, colors, noise_scale=0.01):
    """Create flowing, organic patterns using flow fields"""
    image = Image.new('RGB', (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    
    # Create flow field
    for _ in range(width * height // 1000):  # Adjust density
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        
        points = [(start_x, start_y)]
        x, y = start_x, start_y
        
        # Follow the flow field
        for step in range(100):
            # Create noise-based flow direction
            noise_x = math.sin(x * noise_scale) * math.cos(y * noise_scale)
            noise_y = math.cos(x * noise_scale) * math.sin(y * noise_scale)
            
            x += noise_x * 5
            y += noise_y * 5
            
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
            else:
                break
        
        if len(points) > 2:
            color = random.choice(colors)
            draw.line(points, fill=color, width=random.randint(1, 4))
    
    return image

def create_geometric_tessellation(width, height, colors):
    """Create sophisticated geometric tessellation patterns"""
    image = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # Hexagonal tessellation
    hex_size = min(width, height) // 15
    
    for row in range(-2, height // hex_size + 2):
        for col in range(-2, width // hex_size + 2):
            # Hexagon center
            cx = col * hex_size * 1.5
            cy = row * hex_size * math.sqrt(3)
            
            # Offset every other row
            if row % 2:
                cx += hex_size * 0.75
            
            # Create hexagon points
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                x = cx + hex_size * math.cos(angle)
                y = cy + hex_size * math.sin(angle)
                points.append((x, y))
            
            # Choose color based on position for interesting patterns
            color_idx = (row + col) % len(colors)
            color = colors[color_idx]
            
            # Add some randomness to colors
            if random.random() < 0.3:
                color = random.choice(colors)
            
            draw.polygon(points, fill=color, outline=(0, 0, 0, 50), width=1)
    
    return image

def create_gradient_art(width, height, colors, style="radial"):
    """Create sophisticated gradient artwork"""
    image = Image.new('RGB', (width, height))
    pixels = image.load()
    
    center_x, center_y = width // 2, height // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    for y in range(height):
        for x in range(width):
            if style == "radial":
                # Radial gradient
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                ratio = distance / max_distance
            elif style == "linear":
                # Linear gradient
                ratio = x / width
            elif style == "diagonal":
                # Diagonal gradient
                ratio = (x + y) / (width + height)
            elif style == "wave":
                # Wave pattern
                wave = math.sin(x * 0.01) * math.cos(y * 0.01)
                ratio = (wave + 1) / 2
            
            # Interpolate between colors
            color_idx = ratio * (len(colors) - 1)
            idx1 = int(color_idx)
            idx2 = min(idx1 + 1, len(colors) - 1)
            blend = color_idx - idx1
            
            color1 = colors[idx1]
            color2 = colors[idx2]
            
            r = int(color1[0] * (1 - blend) + color2[0] * blend)
            g = int(color1[1] * (1 - blend) + color2[1] * blend)
            b = int(color1[2] * (1 - blend) + color2[2] * blend)
            
            pixels[x, y] = (r, g, b)
    
    return image

def create_abstract_composition(width, height, colors):
    """Create sophisticated abstract compositions"""
    image = Image.new('RGB', (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    
    # Create multiple layers of shapes
    for layer in range(3):
        layer_opacity = 1.0 - layer * 0.2
        
        for _ in range(5 + layer * 3):
            shape_type = random.choice(['circle', 'organic', 'triangle'])
            color = random.choice(colors)
            
            # Apply opacity
            if len(color) == 3:
                color = color + (int(255 * layer_opacity),)
            
            if shape_type == 'circle':
                radius = random.randint(width//20, width//8)
                x = random.randint(radius, width - radius)
                y = random.randint(radius, height - radius)
                
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=color)
            
            elif shape_type == 'organic':
                x = random.randint(width//8, 7*width//8)
                y = random.randint(height//8, 7*height//8)
                radius = random.randint(width//25, width//10)
                
                create_organic_blob(draw, (x, y), radius, color)
            
            elif shape_type == 'triangle':
                points = []
                center_x = random.randint(width//8, 7*width//8)
                center_y = random.randint(height//8, 7*height//8)
                size = random.randint(width//25, width//10)
                
                for i in range(3):
                    angle = (2 * math.pi * i) / 3 + random.random()
                    x = center_x + size * math.cos(angle)
                    y = center_y + size * math.sin(angle)
                    points.append((x, y))
                
                draw.polygon(points, fill=color)
    
    return image

def add_texture_overlay(image, intensity=0.1):
    """Add subtle texture overlay for more organic feel"""
    width, height = image.size
    
    # Create noise texture
    noise = Image.new('L', (width, height))
    noise_pixels = noise.load()
    
    for y in range(height):
        for x in range(width):
            noise_value = int(random.random() * 255)
            noise_pixels[x, y] = noise_value
    
    # Apply blur to make it more organic
    noise = noise.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Blend with original image
    image = Image.blend(image.convert('RGB'), 
                       Image.merge('RGB', [noise, noise, noise]), 
                       intensity)
    
    return image

def create_print_artwork(size_name="16x20", style="abstract", palette="warm", 
                        add_texture=True, output_path="print_art.png", custom_colors=None):
    """Create high-quality print artwork"""
    
    width, height = PRINT_SIZES[size_name]
    
    # Use custom colors if provided, otherwise use preset palette
    if custom_colors:
        colors = custom_colors
        palette_name = "custom (extracted from image)"
    else:
        colors = get_color_palette(palette)
        palette_name = palette
    
    print(f"Creating {style} artwork in {palette_name} palette...")
    print(f"Colors: {[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors]}")
    print(f"Size: {width}x{height} pixels ({size_name})")
    
    if style == "abstract":
        image = create_abstract_composition(width, height, colors)
    elif style == "mandala":
        image = Image.new('RGB', (width, height), (250, 250, 250))
        create_mandala_pattern(image, (width//2, height//2), 
                             min(width, height)//3, colors)
    elif style == "flow":
        image = create_flow_field(width, height, colors)
    elif style == "geometric":
        image = create_geometric_tessellation(width, height, colors)
    elif style == "gradient_radial":
        image = create_gradient_art(width, height, colors, "radial")
    elif style == "gradient_wave":
        image = create_gradient_art(width, height, colors, "wave")
    
    # Add texture overlay if requested
    if add_texture:
        image = add_texture_overlay(image, intensity=0.05)
    
    # Apply subtle sharpening for print quality
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    
    # Save with maximum quality
    image.save(output_path, "PNG", quality=100, optimize=False)
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Create high-quality print artwork")
    parser.add_argument("--size", choices=list(PRINT_SIZES.keys()), 
                       default="16x20", help="Print size")
    parser.add_argument("--style", 
                       choices=["abstract", "mandala", "flow", "geometric", 
                               "gradient_radial", "gradient_wave"],
                       default="abstract", help="Art style")
    parser.add_argument("--palette", 
                       choices=["warm", "cool", "earth", "monochrome", "jewel"],
                       default="warm", help="Color palette (ignored if --from-image is used)")
    parser.add_argument("--output", default="print_art.png", help="Output filename")
    parser.add_argument("--texture", action="store_true", default=True, 
                       help="Add texture overlay")
    parser.add_argument("--batch", type=int, help="Create multiple variations")
    
    # New color extraction arguments
    parser.add_argument("--from-image", type=str, help="Extract colors from this image file")
    parser.add_argument("--num-colors", type=int, default=5, 
                       help="Number of colors to extract from image (default: 5)")
    parser.add_argument("--enhance-colors", action="store_true", default=True,
                       help="Enhance extracted colors for better artistic results")
    parser.add_argument("--preview-palette", action="store_true", 
                       help="Create a preview of the extracted color palette")
    parser.add_argument("--multiple-images", nargs="+", type=str,
                       help="Extract colors from multiple images and blend them")
    
    args = parser.parse_args()
    
    # Handle color extraction from images
    custom_colors = None
    if args.from_image:
        print(f"🎨 Extracting colors from: {args.from_image}")
        custom_colors = extract_colors_from_image(args.from_image, args.num_colors)
        
        if args.enhance_colors:
            print("✨ Enhancing colors for better artistic results...")
            custom_colors = enhance_color_palette(custom_colors)
        
        print(f"📝 Extracted colors: {[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in custom_colors]}")
        
        if args.preview_palette:
            palette_preview_path = args.output.replace('.png', '_palette.png')
            create_palette_preview(custom_colors, palette_preview_path)
            print(f"👀 Color palette preview saved: {palette_preview_path}")
    
    elif args.multiple_images:
        print(f"🎨 Extracting and blending colors from {len(args.multiple_images)} images...")
        all_colors = []
        
        for img_path in args.multiple_images:
            colors = extract_colors_from_image(img_path, args.num_colors)
            all_colors.extend(colors)
            print(f"   - {img_path}: {len(colors)} colors extracted")
        
        # Reduce to final palette size using clustering
        if len(all_colors) > args.num_colors:
            try:
                img_array = np.array(all_colors)
                kmeans = KMeans(n_clusters=args.num_colors, random_state=42, n_init=10)
                kmeans.fit(img_array)
                custom_colors = [(int(r), int(g), int(b)) for r, g, b in kmeans.cluster_centers_]
            except ImportError:
                # Fallback: just take the most frequent colors
                custom_colors = all_colors[:args.num_colors]
        else:
            custom_colors = all_colors
        
        if args.enhance_colors:
            print("✨ Enhancing blended colors...")
            custom_colors = enhance_color_palette(custom_colors)
        
        print(f"📝 Final blended palette: {[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in custom_colors]}")
        
        if args.preview_palette:
            palette_preview_path = args.output.replace('.png', '_palette.png')
            create_palette_preview(custom_colors, palette_preview_path)
            print(f"👀 Blended palette preview saved: {palette_preview_path}")
    
    if args.batch:
        os.makedirs("print_art_batch", exist_ok=True)
        styles = ["abstract", "mandala", "flow", "geometric", "gradient_radial"]
        palettes = ["warm", "cool", "earth", "jewel"]
        
        for i in range(args.batch):
            style = random.choice(styles)
            palette = random.choice(palettes)
            filename = f"print_art_batch/artwork_{i:03d}_{style}_{palette}.png"
            
            print(f"Creating artwork {i+1}/{args.batch}: {style} in {palette}")
            create_print_artwork(
                size_name=args.size,
                style=style,
                palette=palette,
                add_texture=args.texture,
                output_path=filename,
                custom_colors=custom_colors
            )
    else:
        create_print_artwork(
            size_name=args.size,
            style=args.style,
            palette=args.palette,
            add_texture=args.texture,
            output_path=args.output,
            custom_colors=custom_colors
        )
    
    print(f"✅ High-quality print artwork saved!")
    print(f"📐 Resolution: {PRINT_SIZES[args.size][0]}x{PRINT_SIZES[args.size][1]} (300 DPI)")
    print(f"🖼️  Ready for professional printing!")

if __name__ == "__main__":
    main() 
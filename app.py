#!/usr/bin/env python3
"""
🎨 CUSTOM PRINT ART GENERATOR - DEPLOYMENT OPTIMIZED 🎨
Beautiful GUI for creating personalized artwork that matches your space!
Optimized for free hosting platforms.
"""

import streamlit as st
import os
import sys
import tempfile
import gc
from pathlib import Path
import io
from PIL import Image
import numpy as np
import logging

# Optimize memory usage for deployment
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTHONHASHSEED'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the scripts directory to the path
sys.path.append('scripts')

# Page configuration
st.set_page_config(
    page_title="🎨 Custom Print Art Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import functions with error handling
@st.cache_resource
def load_art_functions():
    """Load art generation functions with caching"""
    try:
        from create_print_art import (
            extract_colors_from_image, 
            enhance_color_palette, 
            create_print_artwork,
            PRINT_SIZES,
            get_color_palette
        )
        return {
            'extract_colors': extract_colors_from_image,
            'enhance_palette': enhance_color_palette,
            'create_artwork': create_print_artwork,
            'print_sizes': PRINT_SIZES,
            'get_palette': get_color_palette
        }
    except ImportError as e:
        st.error(f"Error importing art functions: {e}")
        st.stop()

# Load functions
art_funcs = load_art_functions()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def cleanup_temp_files():
    """Clean up temporary files to save memory"""
    try:
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if file.startswith('tmp') and (file.endswith('.png') or file.endswith('.jpg')):
                try:
                    os.unlink(os.path.join(temp_dir, file))
                except:
                    pass
    except:
        pass

def process_uploaded_image(uploaded_file, num_colors=5):
    """Process uploaded image and extract colors with memory optimization"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract colors
        colors = art_funcs['extract_colors'](tmp_path, num_colors)
        
        # Clean up immediately
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        return colors
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Custom Print Art Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h3>✨ Create Beautiful Print-Ready Art That Matches Your Space!</h3>
        <p>📸 Upload a photo → 🎨 Get custom artwork → 🖼️ Print at 300 DPI quality!</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for options
    with st.sidebar:
        st.header("🎛️ Art Settings")
        
        # Art style selection
        style = st.selectbox(
            "🎨 Art Style",
            ["abstract", "mandala", "flow", "geometric", "gradient_radial", "gradient_wave"],
            help="Choose the artistic style for your custom artwork"
        )
        
        # Print size selection - limit to smaller sizes for free hosting
        size_options = ["8x10", "11x14", "16x20", "square_12"]
        size = st.selectbox(
            "📐 Print Size",
            size_options,
            index=0,  # Default to 8x10 for faster generation
            help="Select the print size (optimized for free hosting)"
        )
        
        # Advanced options
        st.subheader("⚙️ Options")
        
        num_colors = st.slider(
            "🌈 Colors to Extract",
            min_value=3,
            max_value=6,  # Reduced for performance
            value=4,  # Reduced default
            help="How many dominant colors to extract"
        )
        
        enhance_colors = st.checkbox(
            "✨ Enhance Colors",
            value=True,
            help="Boost saturation and contrast"
        )

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Your Image")
        
        # Single image upload (simplified for deployment)
        uploaded_file = st.file_uploader(
            "Upload a photo to extract colors from",
            type=['png', 'jpg', 'jpeg'],
            help="Upload any image and we'll extract colors to create matching artwork!"
        )
        
        # Preset palette option
        st.subheader("🎨 Or Use Preset Palette")
        preset_palette = st.selectbox(
            "Choose a preset color palette",
            ["none", "warm", "cool", "earth", "monochrome", "jewel"],
            help="Skip color extraction and use a curated palette"
        )

    with col2:
        st.header("👀 Preview & Generate")
        
        # Show color palette preview
        custom_colors = None
        palette_name = ""
        
        if uploaded_file:
            try:
                with st.spinner("🎨 Extracting colors..."):
                    custom_colors = process_uploaded_image(uploaded_file, num_colors)
                    
                    if custom_colors and enhance_colors:
                        custom_colors = art_funcs['enhance_palette'](custom_colors)
                    
                    palette_name = f"Extracted from {uploaded_file.name}"
                    
                # Show color swatches
                if custom_colors:
                    st.subheader("🎨 Extracted Colors")
                    cols = st.columns(len(custom_colors))
                    for i, color in enumerate(custom_colors):
                        with cols[i]:
                            swatch = Image.new('RGB', (100, 100), color)
                            st.image(swatch, width=100)
                            st.caption(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
                    
                    st.success(f"✅ {palette_name}")
                    
                    # Show source image (smaller for performance)
                    st.subheader("📸 Source Image")
                    source_image = Image.open(uploaded_file)
                    # Resize for display to save memory
                    source_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                    st.image(source_image, caption="Source image", use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                custom_colors = None
        
        elif preset_palette != "none":
            custom_colors = art_funcs['get_palette'](preset_palette)
            palette_name = f"Preset: {preset_palette.title()}"
            
            st.subheader("🎨 Preset Colors")
            cols = st.columns(len(custom_colors))
            for i, color in enumerate(custom_colors):
                with cols[i]:
                    swatch = Image.new('RGB', (100, 100), color)
                    st.image(swatch, width=100)
                    st.caption(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
            
            st.success(f"✅ {palette_name}")
        
        # Generate artwork button
        if st.button("🎨 Generate Custom Artwork", type="primary"):
            if not (custom_colors):
                st.error("Please upload an image or select a preset palette!")
            else:
                generate_artwork(style, size, custom_colors, palette_name)

def generate_artwork(style, size, custom_colors, palette_name):
    """Generate the custom artwork with memory optimization"""
    
    with st.spinner("🎨 Creating your custom artwork... This may take 30-60 seconds!"):
        try:
            # Clean up before generation
            cleanup_temp_files()
            gc.collect()
            
            # Create output filename
            output_filename = f"custom_art_{style}_{size}.png"
            
            # Use temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, output_filename)
                
                # Generate the artwork
                image = art_funcs['create_artwork'](
                    size_name=size,
                    style=style,
                    palette="custom",
                    add_texture=True,
                    output_path=output_path,
                    custom_colors=custom_colors
                )
                
                # Success message
                st.markdown("""
                <div class="success-box">
                    <h2>🎉 Your Custom Artwork is Ready!</h2>
                    <p>✨ Professional quality • 🖨️ 300 DPI • 📐 Print-ready</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display the generated artwork
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🖼️ Your Custom Artwork")
                    
                    if os.path.exists(output_path):
                        generated_image = Image.open(output_path)
                        # Resize for display to save memory
                        display_image = generated_image.copy()
                        display_image.thumbnail((600, 600), Image.Resampling.LANCZOS)
                        st.image(display_image, caption=f"{style.title()} style in {palette_name} colors")
                        
                        # Download button
                        with open(output_path, "rb") as file:
                            btn = st.download_button(
                                label="📥 Download High-Resolution Artwork",
                                data=file.read(),
                                file_name=output_filename,
                                mime="image/png",
                                type="primary"
                            )
                    else:
                        st.error("Error: Artwork file not found!")
                
                with col2:
                    st.subheader("📋 Artwork Details")
                    
                    # Get dimensions
                    width, height = art_funcs['print_sizes'][size]
                    
                    st.info(f"""
                    **🎨 Style:** {style.title()}  
                    **📐 Size:** {size.replace('_', ' ')} inches  
                    **📏 Resolution:** {width} × {height} pixels  
                    **🖨️ DPI:** 300 (Professional Print Quality)  
                    **🎨 Colors:** {len(custom_colors)}  
                    **✨ Enhanced:** Yes
                    """)
                    
                    st.success("✅ Ready for professional printing!")
                    
                    # Printing tips
                    with st.expander("🖨️ Printing Tips"):
                        st.write("""
                        **Best Results:**
                        - Use high-quality photo paper or canvas
                        - Print at 300 DPI (no scaling needed!)
                        - Consider professional printing services
                        - Frame with UV-protective glass
                        
                        **Recommended Papers:**
                        - Matte photo paper for soft look
                        - Canvas for gallery presentation
                        - Metal prints for modern style
                        """)
            
            # Clean up after generation
            cleanup_temp_files()
            gc.collect()
        
        except Exception as e:
            st.error(f"Error generating artwork: {str(e)}")
            logger.error(f"Generation error: {e}")

# Sidebar additional info
with st.sidebar:
    st.markdown("---")
    st.subheader("💡 How It Works")
    st.write("""
    1. **Upload** a photo of your space
    2. **AI extracts** the dominant colors
    3. **Generate** beautiful matching artwork
    4. **Download** print-ready files (300 DPI)
    5. **Print** and enjoy your custom art!
    """)
    
    st.subheader("🎯 Perfect For")
    st.write("""
    - 🏠 Home decorating
    - 🏢 Office spaces  
    - 🎁 Custom gifts
    - 🏨 Properties
    - 🎨 Art collections
    """)
    
    st.markdown("---")
    st.caption("Made with ❤️ using Python & Streamlit")

if __name__ == "__main__":
    main() 
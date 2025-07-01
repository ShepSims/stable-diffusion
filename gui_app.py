#!/usr/bin/env python3
"""
🎨 CUSTOM PRINT ART GENERATOR 🎨
Beautiful GUI for creating personalized artwork that matches your space!

Upload a photo of your room, and get stunning print-ready art that perfectly matches your colors!
"""

import streamlit as st
import os
import sys
import tempfile
import zipfile
from pathlib import Path
import io
from PIL import Image
import base64

# Add the scripts directory to the path so we can import our functions
sys.path.append('scripts')

try:
    from create_print_art import (
        extract_colors_from_image, 
        enhance_color_palette, 
        create_palette_preview,
        create_print_artwork,
        PRINT_SIZES,
        get_color_palette
    )
except ImportError as e:
    st.error(f"Error importing print art functions: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🎨 Custom Print Art Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
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
    
    .stButton>button {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Custom Print Art Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h3>✨ Create Beautiful Print-Ready Art That Matches Your Space!</h3>
        <p>📸 Upload a photo of your room → 🎨 Get custom artwork in matching colors → 🖼️ Print at 300 DPI quality!</p>
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
        
        # Print size selection
        size = st.selectbox(
            "📐 Print Size",
            list(PRINT_SIZES.keys()),
            index=2,  # Default to 16x20
            help="Select the print size (all at 300 DPI for professional printing)"
        )
        
        # Advanced options
        st.subheader("⚙️ Advanced Options")
        
        num_colors = st.slider(
            "🌈 Number of Colors to Extract",
            min_value=3,
            max_value=8,
            value=5,
            help="How many dominant colors to extract from your image"
        )
        
        enhance_colors = st.checkbox(
            "✨ Enhance Colors",
            value=True,
            help="Boost saturation and contrast for better artistic results"
        )
        
        add_texture = st.checkbox(
            "🎪 Add Texture Overlay",
            value=True,
            help="Add subtle texture for a more organic, artistic feel"
        )

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Your Images")
        
        # Option 1: Single image upload
        st.subheader("🖼️ Single Image")
        uploaded_file = st.file_uploader(
            "Upload a photo to extract colors from (room, decor, inspiration image, etc.)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload any image and we'll extract the dominant colors to create matching artwork!"
        )
        
        # Option 2: Multiple images upload
        st.subheader("🎭 Multiple Images (Advanced)")
        uploaded_files = st.file_uploader(
            "Upload multiple images to blend their colors together",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload 2-5 images to create a blended color palette from all of them!"
        )
        
        # Option 3: Use preset palette
        st.subheader("🎨 Or Use Preset Palette")
        preset_palette = st.selectbox(
            "Choose a preset color palette",
            ["none", "warm", "cool", "earth", "monochrome", "jewel"],
            help="Skip color extraction and use a curated color palette"
        )

    with col2:
        st.header("👀 Preview & Generate")
        
        # Show color palette preview
        if uploaded_file or uploaded_files or preset_palette != "none":
            custom_colors = None
            palette_name = ""
            
            try:
                if uploaded_files and len(uploaded_files) > 1:
                    # Multiple images - blend colors
                    st.subheader("🎨 Blended Color Palette")
                    
                    all_colors = []
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Extract colors
                        colors = extract_colors_from_image(tmp_path, num_colors)
                        all_colors.extend(colors)
                        
                        # Clean up
                        os.unlink(tmp_path)
                    
                    # Reduce to final palette size using simple selection
                    if len(all_colors) > num_colors:
                        # Take evenly spaced colors
                        step = len(all_colors) // num_colors
                        custom_colors = [all_colors[i * step] for i in range(num_colors)]
                    else:
                        custom_colors = all_colors[:num_colors]
                    
                    if enhance_colors:
                        custom_colors = enhance_color_palette(custom_colors)
                    
                    palette_name = f"Blended from {len(uploaded_files)} images"
                
                elif uploaded_file:
                    # Single image
                    st.subheader("🎨 Extracted Color Palette")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extract colors
                    custom_colors = extract_colors_from_image(tmp_path, num_colors)
                    
                    if enhance_colors:
                        custom_colors = enhance_color_palette(custom_colors)
                    
                    palette_name = f"Extracted from {uploaded_file.name}"
                    
                    # Clean up
                    os.unlink(tmp_path)
                
                elif preset_palette != "none":
                    # Preset palette
                    st.subheader("🎨 Preset Color Palette")
                    custom_colors = get_color_palette(preset_palette)
                    palette_name = f"Preset: {preset_palette.title()}"
                
                # Show color swatches
                if custom_colors:
                    cols = st.columns(len(custom_colors))
                    for i, color in enumerate(custom_colors):
                        with cols[i]:
                            # Create a color swatch
                            swatch = Image.new('RGB', (100, 100), color)
                            st.image(swatch, width=100)
                            st.caption(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
                    
                    st.success(f"✅ {palette_name}")
                    
                    # Show image preview if uploaded
                    if uploaded_file:
                        st.subheader("📸 Your Source Image")
                        source_image = Image.open(uploaded_file)
                        st.image(source_image, caption="Source image for color extraction", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                custom_colors = None
        
        # Generate artwork button
        if st.button("🎨 Generate Custom Artwork", type="primary"):
            if not (uploaded_file or uploaded_files or preset_palette != "none"):
                st.error("Please upload an image or select a preset palette!")
            else:
                generate_artwork(style, size, custom_colors, add_texture, palette_name)

def generate_artwork(style, size, custom_colors, add_texture, palette_name):
    """Generate the custom artwork and display results"""
    
    with st.spinner("🎨 Creating your custom artwork... This may take a moment!"):
        try:
            # Create output filename
            output_filename = f"custom_art_{style}_{size}.png"
            output_path = os.path.join("outputs", output_filename)
            
            # Ensure outputs directory exists
            os.makedirs("outputs", exist_ok=True)
            
            # Generate the artwork
            image = create_print_artwork(
                size_name=size,
                style=style,
                palette="custom",  # This will be ignored since we're passing custom_colors
                add_texture=add_texture,
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
                
                # Load and display the generated image
                if os.path.exists(output_path):
                    generated_image = Image.open(output_path)
                    # Resize for display (keep aspect ratio)
                    display_image = generated_image.copy()
                    display_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
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
                width, height = PRINT_SIZES[size]
                
                st.info(f"""
                **🎨 Style:** {style.title()}  
                **📐 Size:** {size.replace('_', ' ')} inches  
                **📏 Resolution:** {width} × {height} pixels  
                **🖨️ DPI:** 300 (Professional Print Quality)  
                **🎨 Colors:** {len(custom_colors) if custom_colors else 'Preset'}  
                **✨ Enhanced:** {'Yes' if custom_colors else 'N/A'}  
                **🎪 Textured:** {'Yes' if add_texture else 'No'}
                """)
                
                st.success("✅ Ready for professional printing!")
                
                # Printing tips
                with st.expander("🖨️ Printing Tips"):
                    st.write("""
                    **Best Results:**
                    - Use high-quality photo paper or canvas
                    - Print at 300 DPI (no scaling needed!)
                    - Consider professional printing services
                    - Frame with UV-protective glass for longevity
                    
                    **Recommended Papers:**
                    - Matte photo paper for soft, artistic look
                    - Glossy paper for vibrant colors
                    - Canvas for gallery-style presentation
                    - Metal prints for modern, sleek appearance
                    """)
        
        except Exception as e:
            st.error(f"Error generating artwork: {str(e)}")
            st.exception(e)

def show_examples():
    """Show example use cases and results"""
    st.header("🌟 Examples & Inspiration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏠 Living Room Match")
        st.write("Upload a photo of your sofa and get matching wall art!")
        
    with col2:
        st.subheader("💼 Office Decor") 
        st.write("Professional artwork that matches your workspace colors!")
        
    with col3:
        st.subheader("🎁 Custom Gifts")
        st.write("Create personalized art based on favorite photos!")

# Sidebar additional info
with st.sidebar:
    st.markdown("---")
    st.subheader("💡 How It Works")
    st.write("""
    1. **Upload** a photo of your room/decor
    2. **AI extracts** the dominant colors
    3. **Generate** beautiful artwork using those colors
    4. **Download** print-ready files (300 DPI)
    5. **Print** and enjoy your custom art!
    """)
    
    st.subheader("🎯 Perfect For")
    st.write("""
    - 🏠 Home decorating
    - 🏢 Office spaces  
    - 🎁 Custom gifts
    - 🏨 Airbnb properties
    - 🎨 Art collections
    """)
    
    st.markdown("---")
    st.write("Made with ❤️ using Python & Streamlit")

if __name__ == "__main__":
    main() 
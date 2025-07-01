#!/usr/bin/env python3
"""
Test script for Stable Diffusion Fine-tuning API
"""

import requests
import json
import time
import os
from pathlib import Path
import argparse

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        print("✅ Health check passed")
        print(json.dumps(response.json(), indent=2))
    else:
        print("❌ Health check failed")
        print(f"Status: {response.status_code}")
        print(response.text)
    
    return response.status_code == 200

def test_upload_images(model_name, image_dir):
    """Test image upload endpoint"""
    print(f"\nTesting image upload for model: {model_name}")
    
    # Get image files from directory
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return False
    
    # Limit to 25 images
    image_files = image_files[:25]
    print(f"Found {len(image_files)} images")
    
    # Prepare files for upload
    files = []
    for image_file in image_files:
        files.append(('files', (image_file.name, open(image_file, 'rb'), 'image/jpeg')))
    
    data = {'model_name': model_name}
    
    try:
        response = requests.post(f"{API_BASE_URL}/upload-training-images", files=files, data=data)
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        if response.status_code == 200:
            print("✅ Image upload successful")
            result = response.json()
            print(f"Uploaded {result['message']}")
            return True
        else:
            print("❌ Image upload failed")
            print(f"Status: {response.status_code}")
            print(response.text)
            return False
    
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return False

def test_start_training(model_name):
    """Test training start endpoint"""
    print(f"\nTesting training start for model: {model_name}")
    
    training_config = {
        "model_name": model_name,
        "training_steps": 500,  # Reduced for testing
        "learning_rate": 1e-4,
        "batch_size": 1,
        "resolution": 512,
        "use_8bit_adam": True,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16"
    }
    
    response = requests.post(f"{API_BASE_URL}/start-training", json=training_config)
    
    if response.status_code == 200:
        print("✅ Training started successfully")
        result = response.json()
        job_id = result['job_id']
        print(f"Job ID: {job_id}")
        return job_id
    else:
        print("❌ Training start failed")
        print(f"Status: {response.status_code}")
        print(response.text)
        return None

def test_training_status(job_id):
    """Test training status endpoint"""
    print(f"\nTesting training status for job: {job_id}")
    
    response = requests.get(f"{API_BASE_URL}/training-status/{job_id}")
    
    if response.status_code == 200:
        print("✅ Training status retrieved successfully")
        result = response.json()
        print(json.dumps(result, indent=2))
        return result
    else:
        print("❌ Training status retrieval failed")
        print(f"Status: {response.status_code}")
        print(response.text)
        return None

def test_list_models():
    """Test model listing endpoint"""
    print("\nTesting model listing...")
    
    response = requests.get(f"{API_BASE_URL}/list-models")
    
    if response.status_code == 200:
        print("✅ Model listing successful")
        result = response.json()
        print(json.dumps(result, indent=2))
        return result
    else:
        print("❌ Model listing failed")
        print(f"Status: {response.status_code}")
        print(response.text)
        return None

def test_generate_image(model_name):
    """Test image generation endpoint"""
    print(f"\nTesting image generation with model: {model_name}")
    
    generation_config = {
        "prompt": "a beautiful landscape in the style",
        "negative_prompt": "blurry, low quality",
        "model_name": model_name,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "num_images": 1,
        "seed": 42
    }
    
    response = requests.post(f"{API_BASE_URL}/generate", json=generation_config)
    
    if response.status_code == 200:
        print("✅ Image generation successful")
        result = response.json()
        print(f"Generated {result['count']} images")
        return True
    else:
        print("❌ Image generation failed")
        print(f"Status: {response.status_code}")
        print(response.text)
        return False

def monitor_training(job_id, timeout=3600):
    """Monitor training progress"""
    print(f"\nMonitoring training job: {job_id}")
    
    start_time = time.time()
    
    while True:
        status = test_training_status(job_id)
        
        if status is None:
            print("❌ Could not get training status")
            break
        
        current_status = status['status']
        progress = status['progress']
        
        print(f"Status: {current_status}, Progress: {progress:.1f}%")
        
        if current_status == 'completed':
            print("✅ Training completed successfully!")
            return True
        elif current_status == 'failed':
            print("❌ Training failed!")
            print(f"Error: {status.get('error_message', 'Unknown error')}")
            return False
        
        # Check timeout
        if time.time() - start_time > timeout:
            print("⏰ Training monitoring timeout reached")
            return False
        
        # Wait before next check
        time.sleep(30)

def main():
    global API_BASE_URL
    
    parser = argparse.ArgumentParser(description="Test Stable Diffusion API")
    parser.add_argument("--model-name", default="test-model", help="Model name for testing")
    parser.add_argument("--image-dir", required=True, help="Directory containing training images")
    parser.add_argument("--full-test", action="store_true", help="Run full test including training")
    parser.add_argument("--api-url", default=API_BASE_URL, help="API base URL")
    
    args = parser.parse_args()
    
    # Update API_BASE_URL if provided
    API_BASE_URL = args.api_url
    
    print(f"Testing API at: {API_BASE_URL}")
    print(f"Model name: {args.model_name}")
    print(f"Image directory: {args.image_dir}")
    
    # Test basic endpoints
    if not test_health_check():
        print("❌ Basic health check failed, stopping tests")
        return
    
    # Test model listing
    test_list_models()
    
    # Test image upload
    if not test_upload_images(args.model_name, args.image_dir):
        print("❌ Image upload failed, stopping tests")
        return
    
    if args.full_test:
        # Test training
        print("\n" + "="*50)
        print("STARTING FULL TRAINING TEST")
        print("="*50)
        
        job_id = test_start_training(args.model_name)
        if job_id:
            # Monitor training
            training_success = monitor_training(job_id)
            
            if training_success:
                # Test generation with trained model
                test_generate_image(args.model_name)
    else:
        print("\nSkipping training test. Use --full-test to include training.")
        print("Testing generation with base model...")
        test_generate_image("base")
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print("✅ All basic tests completed!")

if __name__ == "__main__":
    main()
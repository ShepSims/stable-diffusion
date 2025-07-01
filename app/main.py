import os
import asyncio
import shutil
from typing import List, Optional
from datetime import datetime
import uuid
import zipfile

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles

from .training.trainer import LoRATrainer
from .models.model_manager import ModelManager
from .utils.image_processor import ImageProcessor
from .config import settings

app = FastAPI(
    title="Stable Diffusion Fine-tuning API",
    description="API for fine-tuning Stable Diffusion models on custom styles",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_manager = ModelManager()
image_processor = ImageProcessor()
trainer = LoRATrainer()

class TrainingRequest(BaseModel):
    model_name: str
    training_steps: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 1
    resolution: int = 512
    use_8bit_adam: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"

class TrainingStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_step: int
    total_steps: int
    elapsed_time: Optional[str] = None
    estimated_remaining: Optional[str] = None
    error_message: Optional[str] = None

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    model_name: str
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    num_images: int = 1
    seed: Optional[int] = None

# Global training status tracker
training_jobs = {}

@app.get("/")
async def root():
    return {"message": "Stable Diffusion Fine-tuning API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload-training-images")
async def upload_training_images(
    files: List[UploadFile] = File(...),
    model_name: str = Form(...)
):
    """Upload training images for fine-tuning (accepts up to 25 images)"""
    
    if len(files) > 25:
        raise HTTPException(status_code=400, detail="Maximum 25 images allowed")
    
    # Create directory for this training set
    upload_dir = f"uploads/{model_name}"
    os.makedirs(upload_dir, exist_ok=True)
    
    uploaded_files = []
    
    try:
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
            
            # Generate unique filename
            file_extension = file.filename.split('.')[-1]
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process and validate image
            processed_path = await image_processor.process_training_image(file_path)
            uploaded_files.append({
                "original_filename": file.filename,
                "processed_path": processed_path,
                "size": len(content)
            })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
    
    return {
        "message": f"Successfully uploaded {len(uploaded_files)} images",
        "model_name": model_name,
        "files": uploaded_files
    }

@app.post("/start-training")
async def start_training(
    background_tasks: BackgroundTasks,
    request: TrainingRequest
):
    """Start fine-tuning process"""
    
    # Check if images exist
    upload_dir = f"uploads/{request.model_name}"
    if not os.path.exists(upload_dir):
        raise HTTPException(status_code=400, detail="No training images found. Upload images first.")
    
    image_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) == 0:
        raise HTTPException(status_code=400, detail="No valid training images found")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize training status
    training_jobs[job_id] = TrainingStatus(
        job_id=job_id,
        status="starting",
        progress=0.0,
        current_step=0,
        total_steps=request.training_steps
    )
    
    # Start training in background
    background_tasks.add_task(
        run_training_job,
        job_id,
        request,
        upload_dir
    )
    
    return {
        "job_id": job_id,
        "message": "Training started",
        "status": "starting"
    }

async def run_training_job(job_id: str, request: TrainingRequest, upload_dir: str):
    """Background task to run the training job"""
    try:
        training_jobs[job_id].status = "running"
        
        # Run training
        await trainer.train_lora(
            job_id=job_id,
            model_name=request.model_name,
            training_images_dir=upload_dir,
            training_steps=request.training_steps,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            resolution=request.resolution,
            use_8bit_adam=request.use_8bit_adam,
            gradient_checkpointing=request.gradient_checkpointing,
            mixed_precision=request.mixed_precision,
            progress_callback=lambda step, total: update_training_progress(job_id, step, total)
        )
        
        training_jobs[job_id].status = "completed"
        training_jobs[job_id].progress = 100.0
        
    except Exception as e:
        training_jobs[job_id].status = "failed"
        training_jobs[job_id].error_message = str(e)

def update_training_progress(job_id: str, current_step: int, total_steps: int):
    """Update training progress"""
    if job_id in training_jobs:
        training_jobs[job_id].current_step = current_step
        training_jobs[job_id].total_steps = total_steps
        training_jobs[job_id].progress = (current_step / total_steps) * 100

@app.get("/training-status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[job_id]

@app.get("/list-models")
async def list_models():
    """List all available fine-tuned models"""
    models = model_manager.list_available_models()
    return {"models": models}

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """Generate image using fine-tuned model"""
    try:
        # Generate image using ModelManager
        images = model_manager.generate_image(
            prompt=request.prompt,
            model_name=request.model_name,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            height=request.height,
            width=request.width,
            num_images=request.num_images,
            seed=request.seed
        )
        
        # Save generated images
        output_dir = f"outputs/{request.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for i, image in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}_{i}.png"
            image_path = os.path.join(output_dir, filename)
            image.save(image_path)
            image_paths.append(image_path)
        
        return {
            "message": "Images generated successfully",
            "model_name": request.model_name,
            "prompt": request.prompt,
            "images": image_paths,
            "count": len(images)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")

@app.get("/download-model/{model_name}")
async def download_model(model_name: str):
    """Download trained model"""
    model_path = f"models/{model_name}"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Create zip file
    zip_path = f"{model_path}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, model_path)
                zipf.write(file_path, arcname)
    
    return FileResponse(
        path=zip_path,
        filename=f"{model_name}.zip",
        media_type="application/zip"
    )

@app.delete("/delete-model/{model_name}")
async def delete_model(model_name: str):
    """Delete a fine-tuned model"""
    success = model_manager.delete_model(model_name)
    if success:
        return {"message": f"Model {model_name} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
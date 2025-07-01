# Stable Diffusion Fine-tuning API

A production-ready API for fine-tuning Stable Diffusion models on custom styles using LoRA (Low-Rank Adaptation). Upload up to 25 images, fine-tune a model on your custom style, and generate new images with the learned style.

## 🚀 Features

- **Style Fine-tuning**: Upload 25 images to fine-tune Stable Diffusion on your custom style
- **LoRA Training**: Efficient fine-tuning using Low-Rank Adaptation
- **RESTful API**: Complete API for training management and image generation
- **Background Training**: Non-blocking training with progress monitoring
- **AWS Ready**: Easy deployment to AWS with ECS, ECR, and CloudFormation
- **Docker Support**: Containerized for consistent deployment
- **GPU Optimized**: Supports CUDA with memory optimization
- **Model Management**: List, download, and delete fine-tuned models

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended: RTX 3080, RTX 4080, or better)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for models and training data

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- Docker (for containerized deployment)
- AWS CLI (for AWS deployment)

## 🛠️ Installation

### Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd stable-diffusion
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Create directories**
```bash
mkdir -p models uploads outputs logs
```

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

2. **Or build manually**
```bash
docker build -t stable-diffusion-api .
docker run -p 8000:8000 --gpus all stable-diffusion-api
```

### AWS Deployment

1. **Deploy infrastructure**
```bash
aws cloudformation create-stack \
  --stack-name stable-diffusion-infrastructure \
  --template-body file://aws/cloudformation/infrastructure.yaml \
  --parameters ParameterKey=KeyPairName,ParameterValue=your-key-pair \
  --capabilities CAPABILITY_NAMED_IAM
```

2. **Deploy application**
```bash
cd aws
chmod +x deploy.sh
./deploy.sh
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `DEFAULT_MODEL` | Base Stable Diffusion model | `runwayml/stable-diffusion-v1-5` |
| `DEVICE` | Compute device | `cuda` |
| `MIXED_PRECISION` | Mixed precision training | `fp16` |
| `MAX_TRAINING_IMAGES` | Maximum training images | `25` |
| `DEFAULT_TRAINING_STEPS` | Default training steps | `1000` |
| `AWS_REGION` | AWS region | `us-east-1` |

## 📖 API Usage

### 1. Start the API

```bash
# Local development
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up
```

### 2. Upload Training Images

```bash
curl -X POST "http://localhost:8000/upload-training-images" \
  -F "model_name=my-style" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  ... (up to 25 images)
```

### 3. Start Training

```bash
curl -X POST "http://localhost:8000/start-training" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-style",
    "training_steps": 1000,
    "learning_rate": 1e-4,
    "resolution": 512
  }'
```

### 4. Monitor Training

```bash
curl "http://localhost:8000/training-status/{job_id}"
```

### 5. Generate Images

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape in the style",
    "model_name": "my-style",
    "num_images": 1
  }'
```

## 🎯 API Endpoints

### Training Management
- `POST /upload-training-images` - Upload training images
- `POST /start-training` - Start fine-tuning process
- `GET /training-status/{job_id}` - Get training status
- `GET /list-models` - List available models
- `DELETE /delete-model/{model_name}` - Delete a model

### Image Generation
- `POST /generate` - Generate images with fine-tuned model
- `GET /download-model/{model_name}` - Download model weights

### Utility
- `GET /health` - Health check
- `GET /` - API information

## 🧪 Testing

### Create Sample Images
```bash
python scripts/create_sample_images.py --output-dir sample_images --num-images 25
```

### Test API
```bash
# Basic test
python scripts/test_api.py --image-dir sample_images --model-name test-model

# Full test including training
python scripts/test_api.py --image-dir sample_images --model-name test-model --full-test
```

## 🏗️ Architecture

```
stable-diffusion/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── training/
│   │   └── trainer.py       # LoRA training implementation
│   ├── models/
│   │   └── model_manager.py # Model loading and management
│   └── utils/
│       └── image_processor.py # Image processing utilities
├── aws/                     # AWS deployment files
├── scripts/                 # Utility scripts
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile              # Docker container definition
└── requirements.txt        # Python dependencies
```

## 🔬 Training Details

### LoRA Fine-tuning
- **Method**: Low-Rank Adaptation (LoRA) for efficient fine-tuning
- **Memory Efficient**: Trains only small adapter layers
- **Style Learning**: Learns artistic style from training images
- **Fast Training**: Typically completes in 30-60 minutes on RTX 3080

### Training Parameters
- **Resolution**: 512x512 (configurable)
- **Batch Size**: 1 (memory optimized)
- **Learning Rate**: 1e-4 (configurable)
- **Training Steps**: 1000 (configurable)
- **Mixed Precision**: FP16 for memory efficiency

## 📊 Monitoring

### Training Progress
- Real-time progress tracking via API
- Background training with status updates
- Error handling and logging

### System Monitoring
- Health check endpoint
- Memory usage tracking
- GPU utilization monitoring

## 🚀 Deployment Options

### Local Development
- Direct Python execution
- Docker Compose for local testing

### AWS Production
- **ECS**: Container orchestration
- **ECR**: Container registry
- **ALB**: Load balancing
- **EFS**: Shared storage for models
- **CloudWatch**: Logging and monitoring

### Scaling
- Horizontal scaling with multiple ECS tasks
- Load balancing across instances
- Shared model storage via EFS

## 🛡️ Security

- **Non-root containers**: Docker security best practices
- **Input validation**: File type and size restrictions
- **AWS IAM**: Role-based access control
- **VPC**: Network isolation in AWS

## 📈 Performance

### Optimization Features
- **XFormers**: Memory-efficient attention
- **Gradient Checkpointing**: Reduced memory usage
- **8-bit Adam**: Optimized optimizer
- **CPU Offloading**: GPU memory optimization

### Benchmarks
- **RTX 3080**: ~45 minutes for 1000 steps
- **RTX 4080**: ~30 minutes for 1000 steps
- **Memory Usage**: ~8GB VRAM during training

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size to 1
   - Enable gradient checkpointing
   - Use mixed precision (fp16)

2. **Training Fails**
   - Check image formats (PNG, JPG, JPEG)
   - Ensure minimum 256x256 image size
   - Verify CUDA installation

3. **Slow Training**
   - Enable XFormers if available
   - Use proper GPU instance type
   - Check memory usage

### Debug Commands
```bash
# Check GPU availability
nvidia-smi

# Check CUDA in Python
python -c "import torch; print(torch.cuda.is_available())"

# Monitor training logs
docker-compose logs -f stable-diffusion-api
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Stable Diffusion](https://stability.ai/stable-diffusion)

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Ready to fine-tune your Stable Diffusion models? Start with the installation guide above!** 🎨 
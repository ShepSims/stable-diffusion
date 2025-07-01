#!/bin/bash

# AWS Deployment Script for Stable Diffusion API
set -e

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPOSITORY="${ECR_REPOSITORY:-stable-diffusion-api}"
ECS_CLUSTER="${ECS_CLUSTER:-stable-diffusion-cluster}"
ECS_SERVICE="${ECS_SERVICE:-stable-diffusion-service}"
DOCKER_TAG="${DOCKER_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo_error "Docker is not installed. Please install it first."
    exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$ACCOUNT_ID" ]; then
    echo_error "Unable to get AWS account ID. Please check your AWS credentials."
    exit 1
fi

echo_info "AWS Account ID: $ACCOUNT_ID"
echo_info "Region: $AWS_REGION"

# Create ECR repository if it doesn't exist
echo_info "Creating ECR repository if it doesn't exist..."
aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION 2>/dev/null || \
aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION

# Get ECR login token
echo_info "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build Docker image
echo_info "Building Docker image..."
docker build -t $ECR_REPOSITORY:$DOCKER_TAG .

# Tag image for ECR
docker tag $ECR_REPOSITORY:$DOCKER_TAG $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$DOCKER_TAG

# Push image to ECR
echo_info "Pushing image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$DOCKER_TAG

# Update task definition with new image URI
echo_info "Updating task definition..."
sed "s/ACCOUNT_ID/$ACCOUNT_ID/g; s/REGION/$AWS_REGION/g" aws/ecs-task-definition.json > /tmp/task-definition.json

# Register new task definition
TASK_DEFINITION_ARN=$(aws ecs register-task-definition --cli-input-json file:///tmp/task-definition.json --region $AWS_REGION --query 'taskDefinition.taskDefinitionArn' --output text)

echo_info "New task definition registered: $TASK_DEFINITION_ARN"

# Update ECS service
echo_info "Updating ECS service..."
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --task-definition $TASK_DEFINITION_ARN --region $AWS_REGION

# Wait for deployment to complete
echo_info "Waiting for deployment to complete..."
aws ecs wait services-stable --cluster $ECS_CLUSTER --services $ECS_SERVICE --region $AWS_REGION

echo_info "Deployment completed successfully!"

# Get service details
SERVICE_DETAILS=$(aws ecs describe-services --cluster $ECS_CLUSTER --services $ECS_SERVICE --region $AWS_REGION)
DESIRED_COUNT=$(echo $SERVICE_DETAILS | jq -r '.services[0].desiredCount')
RUNNING_COUNT=$(echo $SERVICE_DETAILS | jq -r '.services[0].runningCount')

echo_info "Service Status: $RUNNING_COUNT/$DESIRED_COUNT tasks running"

# Clean up temporary files
rm -f /tmp/task-definition.json

echo_info "Deployment script completed!"

# Show helpful commands
echo ""
echo "Helpful commands:"
echo "  View service: aws ecs describe-services --cluster $ECS_CLUSTER --services $ECS_SERVICE --region $AWS_REGION"
echo "  View logs: aws logs tail /ecs/stable-diffusion-api --follow --region $AWS_REGION"
echo "  Scale service: aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --desired-count 2 --region $AWS_REGION"
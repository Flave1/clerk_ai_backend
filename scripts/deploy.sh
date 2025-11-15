#!/bin/bash
# Script to build and deploy Clerk Backend to AWS ECS
# Usage: ./scripts/deploy.sh [cluster-name] [service-name]

set -e

AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="588412562130"
ECR_REPOSITORY="clerk_backend"
IMAGE_TAG="stagingv1.0.0"
CLUSTER_NAME="${1:-clerk-cluster}"
SERVICE_NAME="${2:-clerk-backend-service}"

echo "🚀 Starting deployment process..."
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo "Repository: $ECR_REPOSITORY"
echo "Cluster: $CLUSTER_NAME"
echo "Service: $SERVICE_NAME"
echo ""

# Step 1: Login to ECR
echo "📝 Step 1: Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 2: Build Docker image
echo ""
echo "🔨 Step 2: Building Docker image..."
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

# Step 3: Tag image for ECR
echo ""
echo "🏷️  Step 3: Tagging image for ECR..."
ECR_IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_IMAGE_URI}

# Step 4: Push to ECR
echo ""
echo "📤 Step 4: Pushing image to ECR..."
docker push ${ECR_IMAGE_URI}

# Step 5: Register new task definition
echo ""
echo "📋 Step 5: Registering new task definition..."
TASK_DEF_JSON="ecs-task-def.json"
# Update the image URI in the task definition using Python for proper JSON handling
TEMP_TASK_DEF="/tmp/task-def-${IMAGE_TAG}.json"
python3 << EOF > ${TEMP_TASK_DEF}
import json
import sys

with open('${TASK_DEF_JSON}', 'r') as f:
    task_def = json.load(f)

# Update the image URI in the container definition
task_def['containerDefinitions'][0]['image'] = '${ECR_IMAGE_URI}'

print(json.dumps(task_def, indent=2))
EOF

TASK_DEF_ARN=$(aws ecs register-task-definition \
    --cli-input-json file://${TEMP_TASK_DEF} \
    --region $AWS_REGION \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)
# Clean up temp file
rm -f ${TEMP_TASK_DEF}

echo "✅ Task definition registered: $TASK_DEF_ARN"

# Step 6: Update ECS service
echo ""
echo "🔄 Step 6: Updating ECS service..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --task-definition $TASK_DEF_ARN \
    --force-new-deployment \
    --region $AWS_REGION > /dev/null

echo "✅ Service update initiated"

# Step 7: Wait for service to stabilize
echo ""
echo "⏳ Step 7: Waiting for service to stabilize..."
aws ecs wait services-stable \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $AWS_REGION

echo ""
echo "✅ Deployment completed successfully!"

# Step 8: Get public IP address
echo ""
echo "🌐 Step 8: Retrieving public IP address..."
sleep 5  # Give the task a moment to fully initialize

# Get the task ARN
TASK_ARN=$(aws ecs list-tasks \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --region $AWS_REGION \
    --query 'taskArns[0]' \
    --output text)

if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
    # Get the ENI ID from the task
    ENI_ID=$(aws ecs describe-tasks \
        --cluster $CLUSTER_NAME \
        --tasks $TASK_ARN \
        --region $AWS_REGION \
        --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
        --output text)
    
    if [ -n "$ENI_ID" ] && [ "$ENI_ID" != "None" ]; then
        # Get the public IP from the ENI
        PUBLIC_IP=$(aws ec2 describe-network-interfaces \
            --network-interface-ids $ENI_ID \
            --region $AWS_REGION \
            --query 'NetworkInterfaces[0].Association.PublicIp' \
            --output text)
        
        if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "None" ]; then
            echo "✅ Public IP retrieved successfully!"
            echo ""
            echo "📍 Public IP Address: $PUBLIC_IP"
            echo "🔗 Service URL: http://$PUBLIC_IP:8000"
            echo "🏥 Health Check: http://$PUBLIC_IP:8000/health"
        else
            echo "⚠️  Could not retrieve public IP (task may still be initializing)"
        fi
    else
        echo "⚠️  Could not retrieve network interface ID"
    fi
else
    echo "⚠️  Could not retrieve task ARN"
fi

echo ""
echo "📊 Service Status:"
aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $AWS_REGION \
    --query 'services[0].{Status:status,RunningCount:runningCount,DesiredCount:desiredCount}' \
    --output table

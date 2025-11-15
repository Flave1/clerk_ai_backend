#!/bin/bash
# Script to set up Application Load Balancer for ECS Backend
# This creates a stable endpoint for CloudFront to use
# Usage: ./scripts/setup_alb.sh

set -e

AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="588412562130"
CLUSTER_NAME="${CLUSTER_NAME:-clerk-cluster}"
SERVICE_NAME="${SERVICE_NAME:-clerk-backend-service}"
VPC_ID="${VPC_ID:-}"  # Will auto-detect if not set
ALB_NAME="clerk-backend-alb"
TARGET_GROUP_NAME="clerk-backend-tg"
SECURITY_GROUP_NAME="clerk-backend-alb-sg"

echo "🚀 Setting up Application Load Balancer for ECS Backend"
echo "Region: $AWS_REGION"
echo "Cluster: $CLUSTER_NAME"
echo "Service: $SERVICE_NAME"
echo ""

# Step 1: Get VPC and Subnet IDs from ECS service
echo "📋 Step 1: Discovering VPC and subnet configuration..."
SERVICE_DETAILS=$(aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $AWS_REGION \
    --query 'services[0].networkConfiguration.awsvpcConfiguration' \
    --output json)

if [ -z "$VPC_ID" ]; then
    # Get VPC from first subnet
    FIRST_SUBNET=$(echo $SERVICE_DETAILS | python3 -c "import sys, json; print(json.load(sys.stdin)['subnets'][0])")
    VPC_ID=$(aws ec2 describe-subnets \
        --subnet-ids $FIRST_SUBNET \
        --region $AWS_REGION \
        --query 'Subnets[0].VpcId' \
        --output text)
fi

SUBNET_IDS=$(echo $SERVICE_DETAILS | python3 -c "import sys, json; print(','.join(json.load(sys.stdin)['subnets']))")
SUBNET_LIST=($(echo $SUBNET_IDS | tr ',' ' '))

echo "✅ Found VPC: $VPC_ID"
echo "✅ Found Subnets: $SUBNET_IDS"
echo ""

# Step 2: Create Security Group for ALB
echo "🔒 Step 2: Creating security group for ALB..."
ALB_SG_ID=$(aws ec2 create-security-group \
    --group-name $SECURITY_GROUP_NAME \
    --description "Security group for Clerk Backend ALB" \
    --vpc-id $VPC_ID \
    --region $AWS_REGION \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$VPC_ID" \
        --region $AWS_REGION \
        --query 'SecurityGroups[0].GroupId' \
        --output text)

echo "✅ Security Group ID: $ALB_SG_ID"

# Allow HTTP/HTTPS from anywhere (CloudFront will connect)
aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION 2>/dev/null || echo "  (Port 80 rule may already exist)"

aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION 2>/dev/null || echo "  (Port 443 rule may already exist)"

echo "✅ Security group rules configured"
echo ""

# Step 3: Create Target Group
echo "🎯 Step 3: Creating target group..."
TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name $TARGET_GROUP_NAME \
    --protocol HTTP \
    --port 8000 \
    --vpc-id $VPC_ID \
    --target-type ip \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --region $AWS_REGION \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text 2>/dev/null || \
    aws elbv2 describe-target-groups \
        --names $TARGET_GROUP_NAME \
        --region $AWS_REGION \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text)

echo "✅ Target Group ARN: $TARGET_GROUP_ARN"
echo ""

# Step 4: Create Application Load Balancer
echo "⚖️  Step 4: Creating Application Load Balancer..."
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name $ALB_NAME \
    --subnets ${SUBNET_LIST[@]} \
    --security-groups $ALB_SG_ID \
    --scheme internet-facing \
    --type application \
    --ip-address-type ipv4 \
    --region $AWS_REGION \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text 2>/dev/null || \
    aws elbv2 describe-load-balancers \
        --names $ALB_NAME \
        --region $AWS_REGION \
        --query 'LoadBalancers[0].LoadBalancerArn' \
        --output text)

ALB_DNS=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns $ALB_ARN \
    --region $AWS_REGION \
    --query 'LoadBalancers[0].DNSName' \
    --output text)

echo "✅ ALB ARN: $ALB_ARN"
echo "✅ ALB DNS: $ALB_DNS"
echo ""

# Step 5: Create HTTP Listener (redirect to HTTPS)
echo "🔊 Step 5: Creating HTTP listener (redirects to HTTPS)..."
HTTP_LISTENER_ARN=$(aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=redirect,RedirectConfig='{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}' \
    --region $AWS_REGION \
    --query 'Listeners[0].ListenerArn' \
    --output text 2>/dev/null || \
    aws elbv2 describe-listeners \
        --load-balancer-arn $ALB_ARN \
        --region $AWS_REGION \
        --query 'Listeners[?Port==`80`].ListenerArn' \
        --output text | head -1)

echo "✅ HTTP Listener created"
echo ""

# Step 6: Create HTTPS Listener (requires ACM certificate)
echo "🔊 Step 6: Creating HTTPS listener..."
echo "⚠️  Note: You'll need an ACM certificate for api.auray.net"
echo "   Create one in ACM (us-east-1) and update this script with the ARN"
echo ""

# For now, create HTTP listener that forwards to target group
# User can add HTTPS later with ACM certificate
HTTPS_LISTENER_ARN=$(aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN \
    --region $AWS_REGION \
    --query 'Listeners[0].ListenerArn' \
    --output text 2>/dev/null || echo "Listener may already exist")

echo "✅ Listeners configured"
echo ""

# Step 7: Update ECS Service to use ALB
echo "🔄 Step 7: Updating ECS service to use ALB..."
# Get current service configuration
CURRENT_SERVICE=$(aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $AWS_REGION \
    --output json)

# Extract current network config
NETWORK_CONFIG=$(echo $CURRENT_SERVICE | python3 -c "
import sys, json
service = json.load(sys.stdin)['services'][0]
net_config = service.get('networkConfiguration', {})
print(json.dumps(net_config))
")

# Update service with load balancer configuration
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --load-balancers "targetGroupArn=$TARGET_GROUP_ARN,containerName=clerk_backend,containerPort=8000" \
    --network-configuration "$NETWORK_CONFIG" \
    --region $AWS_REGION > /dev/null

echo "✅ ECS service updated to use ALB"
echo ""

# Step 8: Update ECS Task Security Group to allow ALB
echo "🔒 Step 8: Updating ECS task security group..."
# Get current security groups from service
CURRENT_SG=$(echo $SERVICE_DETAILS | python3 -c "import sys, json; print(json.load(sys.stdin)['securityGroups'][0])")

# Allow ALB to reach ECS tasks
aws ec2 authorize-security-group-ingress \
    --group-id $CURRENT_SG \
    --protocol tcp \
    --port 8000 \
    --source-group $ALB_SG_ID \
    --region $AWS_REGION 2>/dev/null || echo "  (Rule may already exist)"

echo "✅ Security group rules updated"
echo ""

# Step 9: Wait for ALB to be active
echo "⏳ Step 9: Waiting for ALB to be active..."
aws elbv2 wait load-balancer-available \
    --load-balancer-arns $ALB_ARN \
    --region $AWS_REGION

echo "✅ ALB is active"
echo ""

# Summary
echo "🎉 ALB Setup Complete!"
echo "=" * 60
echo ""
echo "📋 Configuration Summary:"
echo "  ALB Name: $ALB_NAME"
echo "  ALB DNS: $ALB_DNS"
echo "  ALB ARN: $ALB_ARN"
echo "  Target Group: $TARGET_GROUP_ARN"
echo "  Security Group: $ALB_SG_ID"
echo ""
echo "🔧 Next Steps:"
echo "  1. Update CloudFront origin:"
echo "     - Distribution: E3SV8APOMM7C85 (api.auray.net)"
echo "     - Change origin from 'stagingapi.auray.net' to '$ALB_DNS'"
echo "     - Protocol: HTTP (or HTTPS if you add ACM certificate)"
echo ""
echo "  2. (Optional) Add ACM certificate for HTTPS:"
echo "     - Request certificate in ACM (us-east-1) for api.auray.net"
echo "     - Update HTTPS listener with certificate ARN"
echo ""
echo "  3. Update Squarespace DNS (if using stagingapi):"
echo "     - Point stagingapi.auray.net to $ALB_DNS"
echo ""
echo "  4. Test the endpoint:"
echo "     curl http://$ALB_DNS/health"
echo ""


#!/bin/bash
# Script to add Secrets Manager permissions to ECS Task Execution Role

AWS_REGION="us-east-1"
ROLE_NAME="ecsTaskExecutionRole"

echo "🔧 Adding Secrets Manager permissions to $ROLE_NAME..."

# Create the policy document
cat > /tmp/secrets-manager-policy.json << 'POLICY'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue",
                "secretsmanager:DescribeSecret"
            ],
            "Resource": [
                "arn:aws:secretsmanager:us-east-1:588412562130:secret:clerk/*"
            ]
        }
    ]
}
POLICY

# Attach the policy to the role
aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "SecretsManagerAccess" \
    --policy-document file:///tmp/secrets-manager-policy.json \
    --region $AWS_REGION

if [ $? -eq 0 ]; then
    echo "✅ Successfully added Secrets Manager permissions to $ROLE_NAME"
    echo ""
    echo "The policy allows access to all secrets starting with 'clerk/'"
    rm /tmp/secrets-manager-policy.json
else
    echo "❌ Failed to add permissions"
    exit 1
fi

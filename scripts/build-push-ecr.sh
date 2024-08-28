#!/bin/bash

set -e

ACCOUNT_ID="$1"
REGION="$2"
REPO_NAME="$3"
TAG="$4"

if [ -z "$ACCOUNT_ID" ] || [ -z "$REGION" ] || [ -z "$REPO_NAME" ] || [ -z "$TAG" ]; then
    echo "Usage: $0 <account_id> <region> <repo_name> <tag>"
    exit 1
fi

ECR_URL="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker tag "$REPO_NAME" "$ECR_URL/$REPO_NAME:$TAG"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URL"
aws ecr describe-repositories --repository-names css || aws ecr create-repository --repository-name css
docker push "$ECR_URL/$REPO_NAME:$TAG"

#
# Terraform Init
#
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

#
# Configure the AWS Provider
#
provider "aws" {
  region  = "us-east-1"
}
data "aws_caller_identity" "current" {}

#
# Variables
#
variable "model_name" {
  type = string
}


#
# Outputs
#
output "api_url" {
  value = aws_api_gateway_stage.default.invoke_url
}
output "sagemaker_execution_name" {
  value = aws_iam_role.sagemaker_execution.name
}
output "ecr_url" {
  value = aws_ecr_repository.css.repository_url
}
output "api_token" {
  value     = aws_api_gateway_api_key.css.value
  sensitive = true
}
output "model_name" {
  value = var.model_name
}
output "s3_bucket" {
  value = aws_s3_bucket.css.bucket
}
output "iam_user" {
  value = aws_iam_user.datacloud.arn
}
output "iam_user_access_key" {
  value = aws_iam_access_key.datacloud.id
}
output "iam_user_secret" {
  value     = aws_iam_access_key.datacloud.secret
  sensitive = true
}

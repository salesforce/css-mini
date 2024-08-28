# IAM Policy that allows invoking SageMaker Endpoints
data "aws_iam_policy_document" "invoke" {
  statement {
    actions = [
      "sagemaker:InvokeEndpoint",
    ]
    effect    = "Allow"
    resources = ["*"]
  }
}

# IAM role used by the API Gateway to call Sagemaker
resource "aws_iam_role" "api_sagemaker" {
  name = "css-api-sagemaker"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "apigateway.amazonaws.com"
        }
      },
    ]
  })
  inline_policy {
    name   = "s3"
    policy = data.aws_iam_policy_document.invoke.json
  }
}

# IAM Role used by API Gateway to write logs to CloudWatch
resource "aws_iam_role" "api_cloudwatch" {
  name = "css-api-cloudwatch"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "apigateway.amazonaws.com"
        }
      },
    ]
  })
  managed_policy_arns = ["arn:aws:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs"]
}

# Generic SageMaker execution role
resource "aws_iam_role" "sagemaker_execution" {
  name = "css-sagemaker-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      },
    ]
  })
  managed_policy_arns = ["arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"]
}

# IAM User for Salesforce data cloud
resource "aws_iam_user" "datacloud" {
  name = "datacloud"

}
resource "aws_iam_access_key" "datacloud" {
  user = aws_iam_user.datacloud.name
}
resource "aws_iam_user_policy_attachment" "s3_read" {
  user       = aws_iam_user.datacloud.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

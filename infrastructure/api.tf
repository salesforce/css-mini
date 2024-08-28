# Main API Gateway definition
# This uses OpenAPI to define the paths
resource "aws_api_gateway_rest_api" "sagemaker" {
  name = "css-sagemaker"
  body = jsonencode({
    openapi = "3.0.1"
    info = {
      title   = "sagemaker"
      version = "1.0"
    }
    paths = {
      "/invocations" = {
        post = {
          responses = { 200 = { content = {} } },
          # Require an API key to be used with this path
          security = [{ api_key = [] }]
          # This is the integration with SageMaker
          x-amazon-apigateway-integration = {
            type                = "aws"
            credentials         = aws_iam_role.api_sagemaker.arn
            httpMethod          = "POST"
            uri                 = "arn:aws:apigateway:us-east-1:runtime.sagemaker:path/endpoints/${var.model_name}/invocations"
            responses           = { default = { statusCode = "200" } }
            passthroughBehavior = "when_no_match"
          }
        }
      }
    },
    components = {
      securitySchemes = {
        api_key = {
          type = "apiKey",
          name = "x-api-key",
          in   = "header"
        }
      }
    }
  })

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

# Main deployment of the API
resource "aws_api_gateway_deployment" "default" {
  rest_api_id = aws_api_gateway_rest_api.sagemaker.id

  # Auto deploy when the body of the API definition
  triggers = {
    redeployment = sha1(jsonencode(aws_api_gateway_rest_api.sagemaker.body))
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Default API gateway stage
resource "aws_api_gateway_stage" "default" {
  deployment_id = aws_api_gateway_deployment.default.id
  rest_api_id   = aws_api_gateway_rest_api.sagemaker.id
  stage_name    = "default"
}

# Enable CloudWatch logs for this API
resource "aws_api_gateway_method_settings" "all" {
  rest_api_id = aws_api_gateway_rest_api.sagemaker.id
  stage_name  = aws_api_gateway_stage.default.stage_name
  method_path = "*/*"

  settings {
    metrics_enabled    = false
    logging_level      = "INFO"
    data_trace_enabled = true
  }
}

# Allow API Gateway to write logs
resource "aws_api_gateway_account" "default" {
  cloudwatch_role_arn = aws_iam_role.api_cloudwatch.arn
}

# API Key
resource "aws_api_gateway_api_key" "css" {
  name = "css"
}

# API Usage Plan
resource "aws_api_gateway_usage_plan" "default" {
  name = "css"

  api_stages {
    api_id = aws_api_gateway_rest_api.sagemaker.id
    stage  = aws_api_gateway_stage.default.stage_name
  }
}

resource "aws_api_gateway_usage_plan_key" "css" {
  key_id        = aws_api_gateway_api_key.css.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.default.id
}

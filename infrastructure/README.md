# API Gateway SageMaker Proxy

This terraform module will deploy resources needed to create an API Gateway Proxy for a SageMaker endpoint.

List of resources created:

- IAM
  - Role for API Gateway to allow `sagemaker:InvokeEndpoint`
  - Role for API Gateway to write logs to cloudwatch
  - Role for the SageMaker endpoint
- API Gateway, deployment, and key
- ECS repository to store the model image

## Instructions

[Install terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)

```bash
terraform init
terraform apply
```

After running `terraform apply`, you will be prompted to confirm the deployment.  Enter `yes` to continue.

## Outputs

Terraform will return a few values:

- `api_url`: Base URL for your endpoint
- `ecr_url`: ECR URL for pushing Docker images
- `sagemaker_execution_arn`: ARN for the SageMaker execution role.  Needed when you deploy the endpoint.
- `key_id`: ID of the API key.  See below on how to get the value.

## Get your API key value

Use the following command to get your API key. This will be needed when you connect your model to Salesforce.

```bash
aws apigateway get-api-key --api-key KEY_ID --include-value
```

## Testing the endpoint

You can test your endpoint with this command.

```bash
curl -X POST \
    -H "x-api-key: YOUR_KEY_HERE" \
    -H "Content-Type: application/json" \
    -d '{"instances": [{"features": [0, 1.194449506513182, 0.002951395638504121]}, {"features": [0, 0.160353411548003, 0.13732938785061305]}]}' \
    https://YOUR_URL_HERE/default/invoke
```

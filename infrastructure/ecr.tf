# ECR Repository to store Docker images
resource "aws_ecr_repository" "css" {
  name         = "css-mini"
  force_delete = true
}

# Expire untagged Docker images to avoid excessive storage costs
resource "aws_ecr_lifecycle_policy" "ecr_lifecycle" {
  repository = aws_ecr_repository.css.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Remove untagged images"
      selection = {
        tagStatus   = "untagged"
        countType   = "imageCountMoreThan"
        countNumber = 1
      }
      action = { type = "expire" }
    }]
  })
}

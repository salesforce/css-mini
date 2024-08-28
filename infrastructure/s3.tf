resource "aws_s3_bucket" "css" {
  bucket        = "css-mini-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

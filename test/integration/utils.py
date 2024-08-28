from contextlib import contextmanager
import random
import signal
import string
import subprocess

from botocore.exceptions import ClientError


def build_docker_image(image_uri: str, tag: str, cwd: str):
    """Build a Docker image for the current project."""
    cmd = [
        "docker",
        "build",
        "-t",
        image_uri,
        ".",
    ]
    subprocess.run(cmd, cwd=cwd, check=True)
    return image_uri


class TimeoutError(Exception):
    def __init__(self, limit):
        self.limit = limit
        super().__init__(f"Timed out after {limit} seconds")


@contextmanager
def timeout_serving(seconds, endpoint_name, sagemaker_session):

    def handler(signum, frame):
        raise TimeoutError(seconds)

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        yield

    finally:
        try:
            sagemaker_session.delete_endpoint(endpoint_name)
            sagemaker_session.delete_endpoint_config(endpoint_name)
        except ClientError:
            pass
        signal.alarm(0)


def generate_endpoint_name():
    random_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"css-endpoint-ephemeral-test-{random_id}"


def s3_upload_random(sagemaker_session, data, key_prefix):
    random_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    new_key = f"{key_prefix}/{random_id}"
    return sagemaker_session.upload_data(path=data, key_prefix=new_key)

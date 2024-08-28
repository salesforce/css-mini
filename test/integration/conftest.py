from __future__ import annotations

from pathlib import Path
import shutil
from test.integration.utils import build_docker_image

import pytest
from sagemaker.local import LocalSession

import css

DOCKER_BASE_NAME = "css-mini"


def pytest_addoption(parser):
    parser.addoption("--region")
    parser.addoption("--aws-id")
    parser.addoption("--role")
    parser.addoption("--build-image", "-D", action="store_true")
    parser.addoption("--tag", default=f"{css.__version__}")


@pytest.fixture(scope="module", name="deployment_dir")
def fixture_deployment_dir(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("dummy_deployment")
    tmpdir.mkdir("model_dir")
    tmpdir.mkdir("code_channel")
    tmpdir.mkdir("config_channel")
    tmpdir.mkdir("train_channel")
    yield tmpdir
    shutil.rmtree(str(tmpdir))


@pytest.fixture(name="sagemaker_session")
def fixture_sagemaker_session():
    session = LocalSession()
    session.config = {"local": {"local_code": True}}
    return session


@pytest.fixture(name="build_image", scope="session", autouse=True)
def fixture_build_image(request, tag):
    build_image = request.config.getoption("--build-image")
    repo_root = Path(css.SRC_ROOT).resolve().parent
    image_uri = f"{DOCKER_BASE_NAME}:{tag}"

    if build_image:
        return build_docker_image(
            image_uri=image_uri,
            tag=tag,
            cwd=repo_root,
        )
    return image_uri


@pytest.fixture(name="tag", scope="session")
def fixture_tag(request):
    return request.config.getoption("--tag")


@pytest.fixture(name="region", scope="session")
def fixture_region(request):
    return request.config.getoption("--region")


@pytest.fixture(name="aws_id", scope="session")
def fixture_aws_id(request):
    return request.config.getoption("--aws-id")


@pytest.fixture(name="role", scope="session")
def fixture_role(request):
    return request.config.getoption("--role")


@pytest.fixture(name="ecr_image", scope="session")
def fixture_ecr_image(aws_id, region, tag):
    docker_registry = f"{aws_id}.dkr.ecr.{region}.amazonaws.com"
    return f"{docker_registry}/{DOCKER_BASE_NAME}:{tag}"

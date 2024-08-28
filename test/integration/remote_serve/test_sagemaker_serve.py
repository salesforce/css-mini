import pickle as pkl
import tarfile
from test.integration.utils import (
    generate_endpoint_name,
    s3_upload_random,
    timeout_serving,
)
import uuid

import numpy as np
from numpy.random import PCG64, Generator
import pandas as pd
import pytest
import sagemaker
import scipy.stats as ss

from css import config
from css.serving.utils import pandas_to_datacloud_style_input


@pytest.fixture(name="metric_data", autouse=True, scope="module")
def fixture_metric_data():
    size = 100
    random_gen = Generator(PCG64(2024))
    for dist in (ss.expon, ss.norm, ss.gamma, ss.beta, ss.uniform):
        dist.random_state = random_gen

    acct_ids = pd.Series(
        np.array([str(uuid.uuid4()) for i in range(2 * size // 5)]).repeat(5),
        name="acct_id",
    )
    expon_1 = pd.Series(ss.expon.rvs(loc=0, scale=1.4, size=size), name="metric_expon")
    expon_2 = pd.Series(ss.expon.rvs(loc=0, scale=0.4, size=size), name="metric_expon")
    expon = pd.concat([expon_1, expon_2], ignore_index=True)

    gamma_1 = pd.Series(
        ss.gamma.rvs(a=0.25, loc=0, scale=1.4, size=size), name="metric_gamma"
    )
    gamma_2 = pd.Series(
        ss.gamma.rvs(a=0.75, loc=0, scale=0.4, size=size), name="metric_gamma"
    )
    gamma = pd.concat([gamma_1, gamma_2], ignore_index=True)

    beta_1 = pd.Series(ss.beta.rvs(a=0.25, b=0.25, size=size), name="metric_beta")
    beta_2 = pd.Series(ss.beta.rvs(a=0.75, b=0.75, size=size), name="metric_beta")
    beta = pd.concat([beta_1, beta_2], ignore_index=True)

    norm_1 = pd.Series(ss.norm.rvs(loc=0, scale=1.4, size=size), name="metric_norm")
    norm_2 = pd.Series(ss.norm.rvs(loc=0, scale=0.4, size=size), name="metric_norm")
    norm = pd.concat([norm_1, norm_2], ignore_index=True)

    uniform = pd.Series(ss.uniform.rvs(size=2 * size), name="metric_uni")

    clustering_col = pd.Series([0] * size + [1] * size, name="cluster1")

    return pd.concat(
        [acct_ids, expon, gamma, beta, norm, uniform, clustering_col],
        axis=1,
    )


@pytest.fixture(name="fit_model_dump", autouse=True, scope="module")
def fixture_fit_model_dump(deployment_dir, metric_data):
    default_kwargs = {
        "peer_dims": ["cluster1"],
        "n_neighbors": 50,
    }

    config_model = config.ConfigModel(
        metrics={
            "metric_expon": config.MetricConfigModel(
                type_config_name="ipmexponential", kwargs=default_kwargs
            ),
            "metric_gamma": config.MetricConfigModel(
                type_config_name="ipmgamma", kwargs=default_kwargs
            ),
            "metric_beta": config.MetricConfigModel(
                type_config_name="ipmbeta",
                kwargs=dict(floc=0, fscale=1, **default_kwargs),
            ),
            "metric_norm": config.MetricConfigModel(
                type_config_name="ipmnormal", kwargs=default_kwargs
            ),
        },
        components={
            "comp1": config.ComponentConfigModel(
                metrics=["metric_expon", "metric_gamma"], weights=[1.0, 1.0]
            ),
            "comp2": config.ComponentConfigModel(
                metrics=["metric_beta", "metric_norm"], weights=[1.0, 1.0]
            ),
        },
    )
    model = config_model.to_obj()
    model.fit(metric_data)

    model_dir = deployment_dir / "model_dir"
    with open(model_dir / "css-model", "wb") as f:
        pkl.dump(model, f)
    with tarfile.open(model_dir / "model_data.tar.gz", "w:gz") as tar:
        tar.add(model_dir / "css-model", arcname="css-model")

    return str(model_dir / "model_data.tar.gz")


@pytest.mark.requires_aws
def test_sagemaker_serve(
    role,
    fit_model_dump,
    ecr_image,
    metric_data,
):
    sagemaker_session = sagemaker.Session()
    model_data = s3_upload_random(sagemaker_session, fit_model_dump, "css/test")

    model = sagemaker.model.Model(
        ecr_image,
        role=role,
        sagemaker_session=sagemaker_session,
        model_data=model_data,
        predictor_cls=sagemaker.predictor.Predictor,
    )

    endpoint_name = generate_endpoint_name()
    with timeout_serving(10 * 60, endpoint_name, sagemaker_session):
        predictor = model.deploy(
            instance_type="ml.t2.medium",
            initial_instance_count=1,
            endpoint_name=endpoint_name,
        )
        predictor.serializer = sagemaker.serializers.JSONSerializer()
        predictor.deserializer = sagemaker.deserializers.JSONDeserializer()
        # We use Einstein Studio-style JSON serialization
        min_cols = [
            "cluster1",
            "metric_beta",
            "metric_expon",
            "metric_gamma",
            "metric_norm",
        ]
        min_data = metric_data[min_cols]
        to_pred = pandas_to_datacloud_style_input(min_data)
        pred = predictor.predict(to_pred)
        np.testing.assert_approx_equal(pred["predictions"][0]["global_score"], 4.9)

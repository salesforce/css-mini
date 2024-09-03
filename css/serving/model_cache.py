# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from pathlib import Path
import pickle as pkl
from typing import cast

from css.score import BaseScoringInterface

MODEL_DIR_PATH = Path("/opt/ml/model")


def download_load_model() -> BaseScoringInterface:
    with open(MODEL_DIR_PATH / "css-model", "rb") as f:
        loaded_model = pkl.load(f)
    return loaded_model


class ModelCache:
    _model: BaseScoringInterface | None = None

    @classmethod
    def model(cls) -> BaseScoringInterface:
        if cls._model is None:
            cls.reload_model()
        return cast(BaseScoringInterface, cls._model)

    @classmethod
    def reload_model(cls) -> None:
        cls._model = download_load_model()

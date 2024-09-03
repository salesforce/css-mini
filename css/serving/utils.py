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

from typing import (
    TYPE_CHECKING,
    Literal,
    TypedDict,
    Union,
)

if TYPE_CHECKING:
    import pandas as pd

DataCloudTypes = Union[float, int, str]


class DataCloudInputDict(TypedDict):
    instances: list[dict[Literal["features"], DataCloudTypes]]


_PredictionsInstance = dict[str, DataCloudTypes]


class DataCloudOutputDict(TypedDict):
    predictions: list[_PredictionsInstance]


def pandas_to_datacloud_style_input(df: pd.DataFrame) -> DataCloudInputDict:
    return {"instances": [{"features": v.tolist()} for _, v in df.iterrows()]}


def _tuple_keys_to_str(
    dictionary: dict[tuple[str, str, str], DataCloudTypes]
) -> _PredictionsInstance:
    reform_dict: dict[str, DataCloudTypes] = {}
    for key, value in dictionary.items():
        key_pieces = filter(lambda x: bool(x), key)
        new_key = "_".join(key_pieces)
        reform_dict[new_key] = value
    return reform_dict


def pandas_to_datacloud_style_output(frame: pd.DataFrame) -> DataCloudOutputDict:
    frame_dict = frame.to_dict(orient="index")
    return {"predictions": [_tuple_keys_to_str(v) for _, v in frame_dict.items()]}

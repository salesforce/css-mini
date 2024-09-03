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

from io import StringIO
import json

import pandas as pd

from css.serving import content_types
from css.serving.errors import UnsupportedFormatError


def _csv_to_dataframe(csv: str | bytes) -> pd.DataFrame:
    """Convert CSV data to a DataFrame.

    Simple detection of whether the CSV data contains headers.
    """
    csv_str = csv.decode() if isinstance(csv, bytes) else csv
    maybe_headers = set(csv_str.partition("\n")[0].split(","))
    if all(isinstance(c, str) and not c.isnumeric() for c in maybe_headers):
        # The data obviously contains headers.
        return pd.read_csv(StringIO(csv_str))
    # The data probably does not contain headers.
    return pd.read_csv(StringIO(csv_str), header=None)


def _json_to_dataframe(json_: str | bytes) -> pd.DataFrame:
    """Convert JSON data to a DataFrame.

    Supports both SageMaker and Einstein Studio-style JSON.
    """
    json_str = json_.decode() if isinstance(json_, bytes) else json_
    raw_json = json.loads(json_str)
    if set(raw_json.keys()) == {"instances"}:
        # Received Einstein Studio-style request.
        # Will be of form {"instances": [{"features": [1, 2, 3]}, ...]}
        instances = raw_json["instances"]
        return pd.DataFrame([line["features"] for line in instances])
    # Otherwise, assume the JSON is a single row for prediction (SageMaker-style)
    return pd.DataFrame.from_dict(raw_json, orient="index").T


_decoder_map = {
    content_types.CSV: _csv_to_dataframe,
    content_types.JSON: _json_to_dataframe,
}


def decode(obj: str | bytes, content_type: str) -> pd.DataFrame:
    try:
        decoder = _decoder_map[content_type]
        return decoder(obj)
    except KeyError as exc:
        raise UnsupportedFormatError(content_type) from exc

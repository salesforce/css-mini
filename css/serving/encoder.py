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

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

from css.serving import content_types
from css.serving.errors import UnsupportedFormatError
from css.serving.utils import pandas_to_datacloud_style_output

if TYPE_CHECKING:
    import pandas as pd
    from werkzeug import MIMEAccept


def _tuple_keys_to_str(dictionary: dict[tuple[str, str, str], Any]) -> dict[str, Any]:
    """Flatten nested dictionary coming from Pandas Multi-index."""
    reform_dict: dict[str, Any] = {}
    for key, value in dictionary.items():
        key_pieces = filter(lambda x: x, key)
        new_key = "_".join(key_pieces)
        reform_dict[new_key] = value
    return reform_dict


def _scoring_frame_to_json_bytes(frame: pd.DataFrame) -> bytes:
    reformatted = pandas_to_datacloud_style_output(frame)
    return json.dumps(reformatted).encode()


_encoder_map: dict[str, Callable[[pd.DataFrame], bytes]] = {
    content_types.JSON: _scoring_frame_to_json_bytes,
}


SUPPORTED_CONTENT_TYPES = set(_encoder_map.keys())


def encode(dataframe: pd.DataFrame, accept: MIMEAccept) -> bytes:
    for content_type in accept.values():
        if content_type in SUPPORTED_CONTENT_TYPES:
            try:
                encoder = _encoder_map[content_type]
                return encoder(dataframe)
            except KeyError as exc:
                raise UnsupportedFormatError(content_type) from exc
    raise UnsupportedFormatError(content_type)

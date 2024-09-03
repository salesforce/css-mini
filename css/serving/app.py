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

from flask import Flask, request
import pandas as pd

from css.serving.decoder import decode
from css.serving.encoder import encode
from css.serving.model_cache import ModelCache

app = Flask(__name__)


@app.get("/ping")
def ping():
    """Health check endpoint to ensure the service is operational."""
    return {"message": "ok"}


@app.post("/invocations")
def invocations():
    """Model invoke endpoint."""
    try:
        input_data = decode(request.data, request.content_type)
        model = ModelCache.model()
        if set(model.min_required_columns).issubset(input_data.columns):
            predictions = model.score(input_data)
        elif len(model.min_required_columns) == input_data.shape[1]:
            input_data.columns = model.min_required_columns
            predictions = model.score(input_data)
        else:
            message = (
                f"Model requires columns {model.min_required_columns}, "
                f"but received columns {input_data.columns}. If you pass unlabeled "
                "data, it must match exactly the dimensionality and ordering "
                f"of `min_required_columns`."
            )
            return {"message": message}, 400
    except Exception as e:
        return {"message": str(e)}, 500
    else:
        predictions["last_modified"] = pd.Timestamp.now().isoformat()
        return encode(predictions, request.accept_mimetypes)


def start_server():
    print("Starting Server...")
    app.run(host="0.0.0.0", port=8080)

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

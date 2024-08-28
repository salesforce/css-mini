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

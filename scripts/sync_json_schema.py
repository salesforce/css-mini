import json
from pathlib import Path

from css import SRC_ROOT
from css.config import ConfigModel

JSON_SCHEMA_FILENAME = "config.schema.json"


def main():
    json_schema_path = Path(SRC_ROOT) / JSON_SCHEMA_FILENAME
    json_schema = ConfigModel.model_json_schema()
    with open(json_schema_path, "w") as f:
        f.write(json.dumps(json_schema, indent=2))


if __name__ == "__main__":
    main()

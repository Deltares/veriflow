"""Export config schemas for reference in config files."""

from pathlib import Path

from veriflow.configuration import Config
from veriflow.constants import VERSION

schema_dir = Path(__file__).parent.parent / "schemas" / f"{VERSION}"
schema_path = schema_dir / "config.schema.json"

# Make directory and define path
schema_dir.mkdir(exist_ok=True)

# Write
Config.write_schema(schema_path)

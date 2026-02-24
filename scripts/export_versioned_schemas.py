"""Export config schemas for reference in config files."""

from pathlib import Path

from dpyverification.configuration import Config
from dpyverification.constants import VERSION

schema_dir = Path(__file__).parent.parent / "schemas" / f"{VERSION}"
schema_path = schema_dir / "config.schema.json"

# Check if a folder for this version already exists
if schema_path.exists():
    msg = (
        f"For software version {VERSION} a schema already exists. "
        f"Please ensure no schema exists at: {schema_path}"
    )
    raise ValueError(msg)

# Make directory and define path
schema_dir.mkdir(exist_ok=True)

# Write
Config.write_schema(schema_path)

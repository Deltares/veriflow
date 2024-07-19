"""The schema definition for the configuration yaml file."""

_FEWSWEBSERVICE = {
    "type": "object",
    "required": ["datasourcetype", "url"],
    "properties": {
        "datasourcetype": {"type": "string", "const": "fewswebservice"},
        "url": {"type": "string"},
    },
}

_LOCALFILE = {
    "type": "object",
    "required": ["datasourcetype", "directory", "filename"],
    "properties": {
        "datasourcetype": {"type": "string", "enum": ["pixml", "fewsnetcdf"]},
        "directory": {"type": "string"},
        "filename": {"type": "string"},
    },
}

YAMLSCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "fileversion": {"type": "string"},
        "datasources": {
            "type": "array",
            "minItems": 1,
            "items": {
                "anyOf": [
                    _FEWSWEBSERVICE,
                    _LOCALFILE,
                ],
            },
        },
    },
    "required": ["fileversion", "datasources"],
}

# Config Review Session 

## Config example

See [config.yaml](./config.yaml) at the repository root. It exercises every datasource, score
and datasink currently registered in the v0 schema, with every required and optional option
filled in (optional options are marked with a trailing `#optional` comment). It has been
validated both against `schemas/v0/config.schema.json` and against the actual `Config` /
`ConfigFile` pydantic models.

## Review attention points 

### User experience 
- Names of config fields 
- Description of config fields
- Documentation on website 

### Architecture
- Requirements for `v1.0.0.`?

### Environment variables 
- Is it clear what to set using environment variables? And how to do this?

### ...

## Review findings

### General 

### Datasource config (per supported data source)

#### `BaseDatasource`

#### `FewsWebservice`

#### `FewsNetCDF`



### Score config (per supported score)

#### `BaseScore`
#### `crps_for_ensemble`
#### `rank_histogram`
#### `continuous_scores`
#### `categorical_scores`

### Datasink config (per supported sink)

#### `BaseDatasink`
#### `CFCompliantZarr`
#### `CFCompliantNetCDF`
- **Schema gap**: `CFCompliantZarrConfig` (`cf_compliant_zarr`) and the FEWS NetCDF output
  adapter (`FewsNetCDFOutputConfig`, `fewsnetcdf`) both exist in
  `src/veriflow/configuration/default/datasinks.py`, but `Config.write_schema`'s
  `default_datasinks_config` only includes `CFCompliantNetCDFConfig`. As a result,
  `cf_compliant_zarr`/`fewsnetcdf` datasinks are not recognized by `schemas/v0/config.schema.json`
  and cannot be used in a schema-valid config today, even though the implementation supports
  them. Decide before v1.0.0 whether to register them in the schema or remove/defer the unused
  code.

## Proposed changes 
1. Rename `source_id` to `dataset_id`. 
2. Move `verification_pairs` away from `general`, move to ... in `scores` config section.

## Reviewer notes (Copilot)

Based on reading `src/veriflow/configuration/**`, `src/veriflow/cache/config.py` and the generated
`schemas/v0/config.schema.json` (cross-checked by diffing schema property keys against those with a
`description`). Line numbers are current as of this review and will drift as the code changes.

### Field names — are they clear to users?

- **`import_adapter` / `export_adapter` / `score_adapter`**
  (`src/veriflow/configuration/base.py:217,253,309`) — "adapter" is internal/plugin-architecture
  jargon, not a domain term. Everywhere else the config uses "kind"/"type" for a similar role
  (`data_type`, `spatial_type`, `netcdf_kind`, `ArchiveKind`), so these three discriminator fields
  break that naming pattern and read oddly in plain YAML, e.g. `import_adapter: csv`. Consider
  `datasource_type` / `datasink_type` / `score_type` (or a single consistent `kind`) instead.
- **`source_id`** (`base.py:218`, also `VerificationPair.observations_source_id` /
  `simulations_source_id`) — already flagged above for a rename to `dataset_id`. Agreed: a
  `source_id` reads as "where does this come from" but it is really the identifier of a configured
  *datasource instance* that pairs/id-mapping/scores all reference by string — "dataset id" (or
  "datasource id") describes that role more precisely. Note the rename needs to propagate
  consistently to `observations_source_id`/`simulations_source_id`
  (`configuration/utils.py:208-209`), `IdMap`'s per-source keys, and `StandardAttribute.source_id`
  (`constants.py`), or you end up with inconsistent terminology across the config and the runtime
  dataset attrs.
- **`ThresholdEvent.threshold`** (`configuration/default/scores.py:184`) — this is a *reference to a
  threshold id* defined elsewhere (e.g. `CsvConfig.thresholds`), not an inline numeric threshold.
  Every other cross-reference field in the schema is named `..._id`/`..._ids`
  (`station_ids`, `parameter_ids`, `location_ids`, `verification_pair_ids`); `threshold` alone
  breaks that convention and reads as if a literal value is expected. Suggest `threshold_id`.
- **`thr_factor` / `thr_quantile`** (`configuration/default/scores.py:162,170`) — abbreviated
  ("thr") inconsistently with the rest of the schema, which spells out "threshold" in full
  elsewhere (`ThresholdEvent.threshold`, `CsvConfig.thresholds`). Suggest `threshold_factor` /
  `threshold_quantile`.
- **`LocalFile` vs `LocalFiles`** (`configuration/utils.py:239,246`) — one config element
  (single file: `directory` + `filename`) vs. another (glob of files: `directory` +
  `filename_glob`) differ only by a trailing "s" in the class name, which is an easy source of
  confusion/typos for contributors reading the code (not user-facing, but still worth a clearer
  name, e.g. `LocalFileGlob`).
- **`data_type`** (used on every datasource) — paired conceptually with `spatial_type` (point vs.
  gridded), but `data_type` itself actually encodes the *temporal* characteristic of the series
  (`observed_historical`, `simulated_forecast_ensemble`, etc.) plus `threshold`. The name alone
  doesn't signal that pairing with `spatial_type`, and could be mistaken for a literal dtype.
  Consider a more parallel name, e.g. `temporal_type`.
- **`archive_kind`** (`ArchiveKind`: `open_archive` / `external_storage_archive`,
  `configuration/default/datasources.py:19-23`) — the enum docstring ("Archive kind.") just
  restates the name; a new user has no way to know from the schema what these two options
  actually change in behaviour.

### Structure — is it correct?

- **Hidden, non-schema `general` propagation contract.** `BaseDatasourceConfig`,
  `BaseScoreConfig` and `BaseDatasinkConfig` each declare a *required*
  `general: SkipJsonSchema[GeneralInfoConfig]` field (`base.py:221,291,310`) that is deliberately
  hidden from the JSON schema, and is only ever populated by `ConfigFile.__init__`
  (`configuration/file.py:53-69`), which manually copies `general`/`id_mapping` into every
  datasource/score/sink dict before calling `Config(**yamlcontent)`. We confirmed this is load-bearing:
  calling `Config.model_validate(data)` directly on a parsed YAML dict (without going through
  `ConfigFile`) fails with 12 `general: Field required` errors — one per datasource/score/sink — even
  though `general` is correctly supplied once at the top level. This is a real footgun for anyone
  building a `Config` programmatically (tests already have to special-case it) or integrating
  veriflow's config loading differently, since the contract lives entirely outside the pydantic
  models. Consider moving this propagation into a `Config` validator/constructor so `Config(...)`
  is self-sufficient regardless of how the dict was produced.
- **`extra="allow"` on every config class silently swallows typos.** `BaseConfig.model_config =
  ConfigDict(extra="allow")` (`base.py:206`) is inherited by every datasource/score/datasink
  config, to support user-defined subclasses with extra fields. We verified this also means a
  simple typo (e.g. `stattions:` instead of `stations:`) passes validation silently: the real
  `stations` field stays at its default (`None`) and the misspelled key is just stored as an inert
  extra attribute, with no warning or error anywhere. For a config format that is hand-written
  YAML, this is a significant usability/safety trade-off and a likely source of "my filter didn't
  apply" bug reports. Consider `extra="forbid"` on the shipped default config classes (reserving
  `extra="allow"` only for a documented user-extension base class), so misspelled keys fail fast.
- **`general` mixes unrelated concerns.** It bundles verification scope config
  (`verification_period`, `verification_pairs`, `lead_times`) with an operational/performance
  concern (`cache`, a `ZarrCacheConfig`). This is the same shape of issue already called out for
  `verification_pairs` in "Proposed changes" above — recommend applying the same reasoning to
  `cache` and moving it to its own top-level section (e.g. `cache:` alongside `datasources`/`scores`).
- **`IdMap` is a generic two-level dict with no named structure**
  (`base.py:110-134`, `dict[str, dict[str, str]]`). In the schema it shows up as
  `additionalProperties: {additionalProperties: {type: string}}` with no field names at all, so a
  user can't tell from the schema/autocomplete alone that the outer key is "internal id" and the
  inner key is "source name". A list of explicit records (e.g.
  `{internal_id: discharge, source: observed, external_id: Q.obs}`) would be more verbose but far
  more self-describing and would also pick up per-field descriptions (see below).

### Fields lacking a solid description

Diffing the generated schema's properties against which ones carry a `description` surfaces a
systematic gap (tracked as a TODO in `configuration/config.py:17-23`) plus one likely bug:

- **Bug:** `GeneralInfoConfig.lead_times` (`base.py:46-53`) passes its intended description text as
  the first *positional* argument to `Field(...)` instead of via `description=`. In pydantic that
  positional argument is `default`, so the explanatory text silently never reaches the schema —
  confirmed: `schemas/v0/config.schema.json`'s `lead_times` property has no `description` key at
  all, unlike every sibling field in the same class. Easy fix: add `description=`.
- **No per-field descriptions at all** (only a class-level docstring, which is not surfaced
  per-property in the schema): all of `FewsWebserviceSettings` (`url`, `username`, `password`,
  `max_workers_in_thread_pool`, `archive_kind`, `verify`) and all of `S3AuthConfig`
  (`endpoint_url`, `region_name`, `access_key_id`, `secret_access_key`, `session_token`, `anon`),
  both in `configuration/utils.py`. A user inspecting the schema for just the `url` or
  `access_key_id` property sees no explanation, only the surrounding class docstring (which isn't
  attached to the individual properties when rendered via IDE tooltips/`CTRL+SPACE`).
- **`Range`** (`configuration/utils.py:77-86`: `start`, `end`, `step`) has no description at the
  class or field level anywhere, yet its semantics are non-obvious: `end` is *inclusive*
  (`range(start, end + 1, step)`, see `to_list()`), which is easy to get wrong without reading the
  implementation.
- **`CsvConfig.data_type`** (`configuration/default/datasources.py`, overridden to
  `Literal[DataType.threshold]`) loses the `DataType` enum's description entirely, since a literal
  override doesn't inherit the `$ref`'d enum's description the way `data_type: DataType` does on
  the other datasources.
- **All `*_adapter` discriminator fields** (`import_adapter`, `export_adapter`, `score_adapter`)
  have no description anywhere (see naming note above) — even a one-line "selects which
  datasource/datasink/score implementation this config entry is for" would help.
- **Top-level `Config` properties** (`version`, `general`, `datasources`, `scores`, `datasinks`,
  `id_mapping`, `configuration/config.py:90-95`) have no description directly on the property
  itself; what you see is only the title of the referenced sub-schema, with no explanation of e.g.
  what `datasinks: null` vs. omitted means.
- **Related mismatch worth a description/decision:** `Config.scores` is typed `Sequence[...] | None
  = None` in the Python model (`configuration/config.py:93`), but `Config.write_schema`'s generated
  `ConfigSchema` re-declares `scores` without `| None` (`configuration/config.py:183`), so
  `schemas/v0/config.schema.json` marks `scores` as required (confirmed in the schema's top-level
  `required` list) while the Python model would happily accept a config with `scores` omitted. A
  schema-valid config and a model-valid config are therefore not quite the same set of configs —
  worth a description/decision either way before v1.0.0.

### Environment variables — is it clear what to set, and how?

Two nested settings objects are meant to be sourced from environment variables instead of YAML:
`FewsWebserviceSettings` (`FEWSWEBSERVICE_*`, used via `FewsWebserviceConfig.fews_webservice_settings`)
and `S3AuthConfig` (`S3_*`, used via `BaseZarrConfig.auth_config` on `ZarrConfig`/`ZarrCacheConfig`/
`CFCompliantZarrConfig`). This is documented today only in each class's Python docstring
(`configuration/default/datasources.py:66-108`, `configuration/utils.py:308-337`) — not in the
generated JSON schema (docstrings become the `$defs` entry's `description`, but a user browsing
`fews_webservice_settings`/`auth_config` in an IDE via the YAML schema still sees these as regular
object-typed config fields they could fill in, with no indication that doing so is discouraged in
favour of env vars, and no indication of the required `FEWSWEBSERVICE_URL`/etc. names). We removed
both from [config.yaml](./config.yaml) and left inline comments instead, but that knowledge
shouldn't only live in an example file. Suggested fixes before v1.0.0:
  - Add this guidance to each field's `description=` (not just the class docstring) so it also
    reaches the schema/autocomplete tooltip, e.g. on `fews_webservice_settings` and `auth_config`
    themselves ("Usually left unset; populate via FEWSWEBSERVICE_*/S3_* environment variables
    instead").
  - Document the full list of environment variables once in the user-facing docs (`docs/source/`),
    cross-linked from `CONFIG.md`/the schema descriptions, rather than only in docstrings.

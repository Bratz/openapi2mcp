"""Regenerate finding.schema.json from the Pydantic model.

Usage (from hermes/ with the venv active):  python -m hermes.schemas
Run after ANY change to Finding or its nested models; the unit suite's sync
test fails until you do.
"""

import json
from pathlib import Path

from hermes.schemas.models import Finding

target = Path(__file__).parent / "finding.schema.json"
target.write_text(json.dumps(Finding.model_json_schema(), indent=2) + "\n", encoding="utf-8")
print(f"wrote {target}")

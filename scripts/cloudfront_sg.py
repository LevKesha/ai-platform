"""CloudFront-only ALB security group id. One owner for render and the consumer check."""
from __future__ import annotations

import re

_SG_ID = re.compile(r"sg-[0-9a-f]+")


def require_sg_id(data: dict, key: str) -> str:
    """Return a real sg- id. Missing, blank, or malformed values are errors."""
    if not isinstance(data, dict) or key not in data or data[key] is None:
        raise ValueError(f"{key} is missing or empty")
    raw = data[key]
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be a string")
    sg = raw.strip()
    if not sg:
        raise ValueError(f"{key} is missing or empty")
    if _SG_ID.fullmatch(sg) is None:
        raise ValueError(f"{key} is not a security group id")
    return sg

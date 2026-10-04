"""Internal typing compatibility and fallback helpers for Python < 3.11."""

import sys

if sys.version_info >= (3, 11):
    from typing import (  # type: ignore
        Self,
        Literal,
        TypeAlias,
        get_args,
        get_origin,
    )
else:
    from typing_extensions import (
        Self,
        Literal,
        TypeAlias,
        get_args,
        get_origin,
    )

__all__ = [
    "Self",
    "Literal",
    "TypeAlias",
    "get_args",
    "get_origin",
]

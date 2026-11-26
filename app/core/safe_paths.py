"""Helpers for keeping file access inside an allowed directory."""
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def resolve_within(root: PathLike, candidate: PathLike) -> Optional[Path]:
    """
    Return the fully resolved ``candidate`` if it is ``root`` or lies inside it, else None.

    Both paths are resolved first, so ``..`` segments and symbolic links that point outside
    ``root`` are rejected. A relative ``candidate`` is read relative to ``root``.
    """
    try:
        resolved_root = Path(root).resolve(strict=True)
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = resolved_root / candidate_path
        resolved = candidate_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None

    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved

"""Fail-closed checks for server-local model and cache placement."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


REQUIRED_CACHE_ENV_VARS = (
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HF_HUB_CACHE",
    "HF_TOKEN_PATH",
)


def is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def storage_layout_errors(
    project_root: Path, environment: Mapping[str, str] | None = None
) -> list[str]:
    """Return placement errors without creating or modifying any path."""
    env = os.environ if environment is None else environment
    project = project_root.resolve()
    configured_project = Path(
        env.get("REVISION_FULL_PROJECT_DIR", str(project))
    ).expanduser().resolve()
    storage_root = Path(
        env.get("REVISION_FULL_STORAGE_ROOT", str(project.parent))
    ).expanduser().resolve()
    errors = []
    if configured_project != project:
        errors.append(
            "REVISION_FULL_PROJECT_DIR does not match this checkout: "
            f"{configured_project} != {project}"
        )
    if storage_root != project.parent:
        errors.append(
            "REVISION_FULL_STORAGE_ROOT must be the checkout's immediate parent: "
            f"{storage_root} != {project.parent}"
        )
    if not is_within(project, storage_root):
        errors.append(
            f"project checkout {project} is outside REVISION_FULL_STORAGE_ROOT "
            f"{storage_root}"
        )
    for name in REQUIRED_CACHE_ENV_VARS:
        value = env.get(name)
        if not value:
            errors.append(f"{name} is not set")
            continue
        path = Path(value).expanduser().resolve()
        if not is_within(path, storage_root):
            errors.append(f"{name}={path} is outside {storage_root}")
    return errors


def require_managed_storage(project_root: Path) -> None:
    errors = storage_layout_errors(project_root)
    if errors:
        detail = "\n- ".join(errors)
        raise RuntimeError(
            "Refusing to download outside the configured experiment storage root:\n- "
            f"{detail}\nSource the project server_env.sh in online-staging mode first."
        )

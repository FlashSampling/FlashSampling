"""Content-addressed build identities for the CUTLASS extensions."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_FINGERPRINT_SCHEMA_VERSION = 1
_EXTERNAL_QUOTED_INCLUDE_PREFIXES = ("cute/", "cutlass/")


@dataclass(frozen=True)
class ExtensionBuildSpec:
    """Every local and external input that identifies one extension build."""

    prefix: str
    source_root: Path
    sources: tuple[Path, ...]
    cuda_flags: tuple[str, ...]
    architecture: str
    toolchain_identity: str
    python_abi: str
    torch_version: str
    cuda_version: str
    supplemental_inputs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ExtensionFingerprint:
    """A deterministic digest and the repo-local files that produced it."""

    digest: str
    dependencies: tuple[str, ...]


def extension_fingerprint(spec: ExtensionBuildSpec) -> ExtensionFingerprint:
    """Hash the recursive local include closure and all non-source inputs."""
    root = spec.source_root.resolve()
    dependencies = discover_local_dependencies(root, spec.sources)
    supplemental = tuple(_relative_input(root, path) for path in spec.supplemental_inputs)
    file_inputs = tuple(sorted({*dependencies, *supplemental}))
    metadata = {
        "schema_version": _FINGERPRINT_SCHEMA_VERSION,
        "prefix": spec.prefix,
        "architecture": spec.architecture,
        "cuda_flags": list(spec.cuda_flags),
        "toolchain_identity": spec.toolchain_identity,
        "python_abi": spec.python_abi,
        "torch_version": spec.torch_version,
        "cuda_version": spec.cuda_version,
        "files": list(file_inputs),
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for relative_path in file_inputs:
        path = root / relative_path
        digest.update(relative_path.encode())
        digest.update(path.read_bytes())
    return ExtensionFingerprint(digest.hexdigest(), file_inputs)


def extension_name(spec: ExtensionBuildSpec) -> tuple[str, ExtensionFingerprint]:
    """Return the content-addressed extension name and its audit record."""
    fingerprint = extension_fingerprint(spec)
    return f"{spec.prefix}_{fingerprint.digest[:12]}", fingerprint


def discover_local_dependencies(source_root: Path, sources: tuple[Path, ...]) -> tuple[str, ...]:
    """Resolve quoted includes recursively, rejecting missing local inputs."""
    root = source_root.resolve()
    pending = [path.resolve() for path in sources]
    discovered: set[Path] = set()
    while pending:
        path = pending.pop()
        _require_within_root(root, path)
        if path in discovered:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"CUTLASS build input does not exist: {path}")
        discovered.add(path)
        contents = path.read_text()
        for include in _LOCAL_INCLUDE.findall(contents):
            if include.startswith(_EXTERNAL_QUOTED_INCLUDE_PREFIXES):
                continue
            included_path = _resolve_local_include(root, path.parent, include)
            pending.append(included_path)
    return tuple(sorted(str(path.relative_to(root)) for path in discovered))


def broad_source_fingerprint(source_root: Path) -> ExtensionFingerprint:
    """Reproduce the legacy all-files input set for migration evidence."""
    root = source_root.resolve()
    dependencies = tuple(
        path.name for path in sorted(root.iterdir()) if path.suffix in {".cu", ".cuh", ".patch"}
    )
    digest = hashlib.sha256()
    for relative_path in dependencies:
        digest.update(relative_path.encode())
        digest.update((root / relative_path).read_bytes())
    return ExtensionFingerprint(digest.hexdigest(), dependencies)


def _resolve_local_include(root: Path, parent: Path, include: str) -> Path:
    candidates = (parent / include, root / include)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            _require_within_root(root, resolved)
            return resolved
    raise FileNotFoundError(f'Unresolved quoted CUTLASS include "{include}" from {parent}')


def _relative_input(root: Path, path: Path) -> str:
    resolved = path.resolve()
    _require_within_root(root, resolved)
    if not resolved.is_file():
        raise FileNotFoundError(f"CUTLASS supplemental build input does not exist: {path}")
    return str(resolved.relative_to(root))


def _require_within_root(root: Path, path: Path) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"CUTLASS build input escapes source root: {path}")

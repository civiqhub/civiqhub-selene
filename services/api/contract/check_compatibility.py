# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Fail when a new ``openapi.yaml`` breaks clients of the approved one. Standard library and PyYAML.

Usage, with the contract of the approved base and the contract of the candidate::

    python services/api/contract/check_compatibility.py BASE/openapi.yaml HEAD/openapi.yaml

The CI takes the base from the base branch of a pull request, or from the commit before a
push, never from the candidate: a candidate cannot rewrite its own reference. It checks
the compatibility policy of ``services/api/README.md``:

- the major version stays, and the version grows whenever the document changes;
- every operation and every status code of the base remains, and security is not dropped;
- a request schema accepts everything the base accepted: no property removed, no new
  required property, no narrower type, enum, bound, length or pattern;
- a response schema keeps what clients read: no required property removed, no wider
  type, no new value in an enum, no changed constant.

It reads schemas by component name, as both documents define them.
"""

import json
import sys
from collections.abc import Iterator, Mapping
from typing import Any

import yaml

__all__ = ["breaking_changes", "main"]

METHODS = ("get", "put", "post", "delete", "patch", "head", "options")
_LOWER_BOUNDS = ("minimum", "exclusiveMinimum", "minItems", "minLength", "minProperties")
_UPPER_BOUNDS = ("maximum", "exclusiveMaximum", "maxItems", "maxLength", "maxProperties")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _types(schema: Mapping[str, Any]) -> set[str] | None:
    kind = schema.get("type")
    if kind is None:
        return None
    return {kind} if isinstance(kind, str) else set(kind)


def _ref(schema: Mapping[str, Any]) -> str | None:
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/components/schemas/"):
        return reference.rsplit("/", 1)[-1]
    return None


def _references(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        name = _ref(node)
        if name:
            yield name
        for value in node.values():
            yield from _references(value)
    elif isinstance(node, list):
        for value in node:
            yield from _references(value)


def _reachable(document: Mapping[str, Any], roots: list[object]) -> set[str]:
    schemas = document.get("components", {}).get("schemas", {})
    seen: set[str] = set()
    pending = [name for root in roots for name in _references(root)]
    while pending:
        name = pending.pop()
        if name in seen or name not in schemas:
            continue
        seen.add(name)
        pending.extend(_references(schemas[name]))
    return seen


def _operations(document: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (path, method): operation
        for path, item in document.get("paths", {}).items()
        for method, operation in item.items()
        if method in METHODS
    }


def _responses(document: Mapping[str, Any], operation: Mapping[str, Any]) -> list[object]:
    named = document.get("components", {}).get("responses", {})
    found: list[object] = []
    for response in operation.get("responses", {}).values():
        name = response.get("$ref", "").rsplit("/", 1)[-1] if "$ref" in response else None
        found.append(named.get(name, {}) if name else response)
    return found


def _request(base: Any, head: Any, where: str, problems: list[str]) -> None:
    """Check that ``head`` accepts every request that ``base`` accepted."""
    if not isinstance(base, dict) or not isinstance(head, dict):
        return
    if _ref(base) or _ref(head):
        if _ref(base) != _ref(head):
            problems.append(f"{where}: refers to another schema")
        return
    base_types, head_types = _types(base), _types(head)
    if base_types is not None and head_types is not None and not base_types <= head_types:
        problems.append(f"{where}: accepts fewer types")
    if "enum" in base:
        allowed: list[Any] = head.get("enum") or base["enum"]
        if not set(map(json.dumps, base["enum"])) <= set(map(json.dumps, allowed)):
            problems.append(f"{where}: accepts fewer values")
    if "const" in head and head.get("const") != base.get("const"):
        problems.append(f"{where}: accepts fewer values")
    for bound in _LOWER_BOUNDS:
        if bound in head and (bound not in base or head[bound] > base[bound]):
            problems.append(f"{where}: {bound} is stricter")
    for bound in _UPPER_BOUNDS:
        if bound in head and (bound not in base or head[bound] < base[bound]):
            problems.append(f"{where}: {bound} is stricter")
    if head.get("pattern") != base.get("pattern") and "pattern" in head:
        problems.append(f"{where}: the pattern changed")
    new_required = set(head.get("required", [])) - set(base.get("required", []))
    if new_required:
        problems.append(f"{where}: newly required {sorted(new_required)}")
    if (
        base.get("additionalProperties", True) is not False
        and head.get("additionalProperties") is False
    ):
        problems.append(f"{where}: refuses unknown fields it accepted")
    for name, schema in base.get("properties", {}).items():
        if name not in head.get("properties", {}):
            problems.append(f"{where}.{name}: removed")
        else:
            _request(schema, head["properties"][name], f"{where}.{name}", problems)
    for key in ("items", "additionalProperties"):
        if isinstance(base.get(key), dict) and isinstance(head.get(key), dict):
            _request(base[key], head[key], f"{where}[]", problems)
    for key in ("prefixItems", "oneOf", "anyOf", "allOf"):
        base_list, head_list = base.get(key, []), head.get(key, [])
        if len(head_list) < len(base_list):
            problems.append(f"{where}: fewer {key}")
        for index, (one, other) in enumerate(zip(base_list, head_list, strict=False)):
            _request(one, other, f"{where}.{key}[{index}]", problems)


def _response(base: Any, head: Any, where: str, problems: list[str]) -> None:
    """Check that a client of ``base`` still reads every response of ``head``."""
    if not isinstance(base, dict) or not isinstance(head, dict):
        return
    if _ref(base) or _ref(head):
        if _ref(base) != _ref(head):
            problems.append(f"{where}: refers to another schema")
        return
    base_types, head_types = _types(base), _types(head)
    if base_types is not None and (head_types is None or not head_types <= base_types):
        problems.append(f"{where}: may hold another type")
    if "enum" in base and not set(map(json.dumps, head.get("enum", []))) <= set(
        map(json.dumps, base["enum"])
    ):
        problems.append(f"{where}: may hold a new value")
    if "const" in base and head.get("const") != base["const"]:
        problems.append(f"{where}: the constant changed")
    removed = set(base.get("required", [])) - set(head.get("required", []))
    if removed:
        problems.append(f"{where}: no longer always present {sorted(removed)}")
    for name, schema in base.get("properties", {}).items():
        if name in head.get("properties", {}):
            _response(schema, head["properties"][name], f"{where}.{name}", problems)
        elif name in base.get("required", []):
            problems.append(f"{where}.{name}: removed")
    for key in ("items", "additionalProperties"):
        if isinstance(base.get(key), dict) and isinstance(head.get(key), dict):
            _response(base[key], head[key], f"{where}[]", problems)
    for key in ("prefixItems", "oneOf", "anyOf", "allOf"):
        base_list, head_list = base.get(key, []), head.get(key, [])
        for index, (one, other) in enumerate(zip(base_list, head_list, strict=False)):
            _response(one, other, f"{where}.{key}[{index}]", problems)
        if key in {"oneOf", "anyOf"} and len(head_list) < len(base_list):
            problems.append(f"{where}: fewer {key}")


def breaking_changes(base: Mapping[str, Any], head: Mapping[str, Any]) -> list[str]:
    """Return why ``head`` breaks clients of ``base``; empty when it is compatible."""
    problems: list[str] = []
    base_version, head_version = base["info"]["version"], head["info"]["version"]
    if _version(head_version)[0] != _version(base_version)[0]:
        problems.append(f"info.version: major {head_version} instead of {base_version}")
    if _version(head_version) < _version(base_version):
        problems.append(f"info.version: {head_version} is older than {base_version}")
    unchanged = {**base, "info": None} == {**head, "info": None}
    if not unchanged and _version(head_version) <= _version(base_version):
        problems.append(f"info.version: the document changed but stays {head_version}")

    base_operations, head_operations = _operations(base), _operations(head)
    for key, operation in base_operations.items():
        where = f"{key[1].upper()} {key[0]}"
        if key not in head_operations:
            problems.append(f"{where}: removed")
            continue
        new = head_operations[key]
        missing = set(operation.get("responses", {})) - set(new.get("responses", {}))
        if missing:
            problems.append(f"{where}: no longer answers {sorted(missing)}")
        if operation.get("security", base.get("security")) and not new.get(
            "security", head.get("security")
        ):
            problems.append(f"{where}: no longer requires authentication")

    request_roots = [op.get("requestBody", {}) for op in base_operations.values()]
    request_roots += [op.get("parameters", []) for op in base_operations.values()]
    request_roots += [item.get("parameters", []) for item in base.get("paths", {}).values()]
    response_roots: list[object] = [
        response for op in base_operations.values() for response in _responses(base, op)
    ]
    base_schemas = base.get("components", {}).get("schemas", {})
    head_schemas = head.get("components", {}).get("schemas", {})
    for name in sorted(_reachable(base, request_roots) | _reachable(base, response_roots)):
        if name not in head_schemas:
            problems.append(f"schema {name}: removed")
    for name in sorted(_reachable(base, request_roots)):
        if name in head_schemas:
            _request(base_schemas[name], head_schemas[name], name, problems)
    for name in sorted(_reachable(base, response_roots)):
        if name in head_schemas:
            _response(base_schemas[name], head_schemas[name], name, problems)
    return problems


def main(argv: list[str] | None = None) -> int:
    """Compare two contract files; exit 1 and list the breaking changes, if any."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        print("usage: check_compatibility.py BASE/openapi.yaml HEAD/openapi.yaml", file=sys.stderr)
        return 2
    with open(arguments[0], encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    with open(arguments[1], encoding="utf-8") as stream:
        head = yaml.safe_load(stream)
    if not base.get("paths"):
        print("The base defines no operation yet: there is no approved contract to keep.")
        return 0
    problems = breaking_changes(base, head)
    for problem in problems:
        print(f"breaking change: {problem}")
    if not problems:
        print(
            f"Contract {head['info']['version']} keeps every promise of {base['info']['version']}."
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
